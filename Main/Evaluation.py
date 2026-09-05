import json
import os
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. PARAMÈTRES GLOBAX D'ÉVALUATION
# ==========================================

# Seuil de tolérance pour l'évaluation des synonymes (Soft Match)
SIMILARITY_THRESHOLD = 0.50

# ==========================================

def get_triples_from_graph_dict(graph_data, relation_filter=None, is_prediction=False, evaluate_without_rocade=False):
    """
    Extrait un set de triplets (Source, Relation, Cible) à partir d'un dictionnaire.
    Gère dynamiquement l'ignorance des noms de relations si on teste l'impact "Sans ROCADE".
    """
    node_map = {node["id"]: node["labels"][0].lower().strip() for node in graph_data.get("nodes", [])}
    triples = set()

    for edge in graph_data.get("edges", []):
        src_label = node_map.get(edge.get("source"), "")
        tgt_label = node_map.get(edge.get("target"), "")
        rel_type = edge.get("relation_type", "").upper()

        if relation_filter:
            if evaluate_without_rocade and is_prediction:
                # En mode "Sans ROCADE", le LLM invente des noms temporels (ex: LEADS_TO).
                # On ne peut donc pas filtrer la prédiction par "BEFORE". On garde toute la structure.
                pass
            elif rel_type != relation_filter.upper():
                continue

        # Si on est en mode "Sans ROCADE", on remplace le nom par un joker "*" (Structure seule)
        final_rel = "*" if evaluate_without_rocade else rel_type

        if src_label and tgt_label:
            triples.add((src_label, final_rel, tgt_label))

    return triples

def calculate_independent_metrics(gt_triples, pred_triples, model, similarity_threshold, evaluate_without_rocade=False):
    """
    Calcule les TP, FP, FN pour un seul graphe avec un Soft Match optimisé (tri par similarité).
    """
    if len(gt_triples) == 0:
        return 0, len(pred_triples), 0

    # --- 1. EXACT MATCH ---
    exact_tp_set = gt_triples.intersection(pred_triples)
    exact_tp = len(exact_tp_set)
    remaining_pred = list(pred_triples - exact_tp_set)
    remaining_gt = list(gt_triples - exact_tp_set)

    # --- 2. SOFT MATCH OPTIMISÉ (Anti-Vol) ---
    emb_cache = {}
    def get_embedding(text):
        if text not in emb_cache:
            emb_cache[text] = model.encode(text, convert_to_tensor=True)
        return emb_cache[text]

    potential_matches = []

    for p_idx, (p_src, p_rel, p_tgt) in enumerate(remaining_pred):
        p_src_emb = get_embedding(p_src)
        p_tgt_emb = get_embedding(p_tgt)

        for g_idx, (g_src, g_rel, g_tgt) in enumerate(remaining_gt):
            # Si evaluate_without_rocade est True, p_rel et g_rel valent "*", donc la condition est validée.
            if p_rel != g_rel:
                continue

            g_src_emb = get_embedding(g_src)
            g_tgt_emb = get_embedding(g_tgt)

            sim_src = util.cos_sim(p_src_emb, g_src_emb).item()
            sim_tgt = util.cos_sim(p_tgt_emb, g_tgt_emb).item()

            if sim_src >= similarity_threshold and sim_tgt >= similarity_threshold:
                avg_sim = (sim_src + sim_tgt) / 2
                potential_matches.append((avg_sim, p_idx, g_idx))

    # Tri décroissant pour s'assurer que les meilleures correspondances sont validées en premier
    potential_matches.sort(key=lambda x: x[0], reverse=True)

    soft_tp = 0
    matched_gt_indices = set()
    matched_pred_indices = set()

    for sim, p_idx, g_idx in potential_matches:
        if p_idx not in matched_pred_indices and g_idx not in matched_gt_indices:
            soft_tp += 1
            matched_pred_indices.add(p_idx)
            matched_gt_indices.add(g_idx)

    # --- 3. RETOUR DES COMPTEURS ---
    total_tp = exact_tp + soft_tp
    soft_fp = len(pred_triples) - total_tp
    soft_fn = len(gt_triples) - total_tp

    return total_tp, soft_fp, soft_fn

def print_micro_average_metrics(title, total_tp, total_fp, total_fn, threshold):
    """Affiche les résultats finaux."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    s_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    s_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    s_f1 = 2 * (s_prec * s_rec) / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    print(f"[MÉTRIQUES MICRO-AVERAGE (Seuil: {threshold})]")
    print(f"- Vrais Positifs (TP)  : {total_tp}")
    print(f"- Faux Positifs (FP)   : {total_fp}")
    print(f"- Faux Négatifs (FN)   : {total_fn}")
    print("-" * 30)
    print(f"Précision : {s_prec:.4f} | Rappel : {s_rec:.4f} | F1-Score : {s_f1:.4f}\n")


def evaluate_micro_average(evaluation_pairs: list, similarity_threshold):
    """
    Évalue indépendamment chaque rapport puis calcule la moyenne (Micro-Average).
    Vérifie automatiquement si 'Rocade_False' est présent dans le nom du fichier prédit
    pour activer l'évaluation sans ROCADE.
    """
    print(f"\n[*] Chargement du modèle de similarité sémantique (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compteurs globaux
    global_tp, global_fp, global_fn = 0, 0, 0
    temp_tp, temp_fp, temp_fn = 0, 0, 0

    for pair in evaluation_pairs:
        gold_path = pair["gold"]
        pred_path = pair["pred"]

        if not os.path.exists(gold_path) or not os.path.exists(pred_path):
            print(f"[!] Fichier manquant ignoré : {gold_path} ou {pred_path}")
            continue

        # Détection automatique de ROCADE d'après le nom du fichier de prédiction
        evaluate_without_rocade = "Rocade_False" in pred_path

        with open(gold_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        # 1. Évaluation Globale pour ce rapport
        gt_all = get_triples_from_graph_dict(gt_data, is_prediction=False, evaluate_without_rocade=evaluate_without_rocade)
        pred_all = get_triples_from_graph_dict(pred_data, is_prediction=True, evaluate_without_rocade=evaluate_without_rocade)
        tp, fp, fn = calculate_independent_metrics(gt_all, pred_all, model, similarity_threshold, evaluate_without_rocade=evaluate_without_rocade)
        global_tp += tp; global_fp += fp; global_fn += fn

        # 2. Évaluation Temporelle pour ce rapport
        gt_temp = get_triples_from_graph_dict(gt_data, relation_filter="BEFORE", is_prediction=False, evaluate_without_rocade=evaluate_without_rocade)
        pred_temp = get_triples_from_graph_dict(pred_data, relation_filter="BEFORE", is_prediction=True, evaluate_without_rocade=evaluate_without_rocade)
        t_tp, t_fp, t_fn = calculate_independent_metrics(gt_temp, pred_temp, model, similarity_threshold, evaluate_without_rocade=evaluate_without_rocade)
        temp_tp += t_tp; temp_fp += t_fp; temp_fn += t_fn

    # Affichage du mode global de la série
    mode_str = "SANS ROCADE (Évaluation Structurelle pure)" if "Rocade_False" in evaluation_pairs[0]["pred"] else "AVEC ROCADE (Évaluation Sémantique stricte)"
    print(f"\n[*] Mode appliqué pour cette série : {mode_str}")

    # Affichage des résultats consolidés
    print_micro_average_metrics("1. ÉVALUATION GLOBALE (MICRO-AVERAGE)", global_tp, global_fp, global_fn, similarity_threshold)
    print_micro_average_metrics("2. ÉVALUATION TEMPORELLE (MICRO-AVERAGE 'BEFORE')", temp_tp, temp_fp, temp_fn, similarity_threshold)


if __name__ == "__main__":
    import os

    # Renseignez ici le chemin de base absolu de votre projet
    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"

    print("\n" + "="*80)
    print("LANCEMENT DE LA CAMPAGNE D'ÉVALUATION COMPLÈTE (8 USE CASES)")
    print("="*80)

    # 3 boucles imbriquées pour générer automatiquement les 2x2x2 = 8 configurations
    for test_rocade in [True, False]:
        for test_chunking in [True, False]:
            for test_prompt in ["cot", "few_shot"]:

                # Construction automatique du suffixe exact du fichier
                file_suffix = f"Rocade_{test_rocade}_Chunking_{test_chunking}_{test_prompt}.json"

                EVALUATION_PAIRS = [
                    {
                        "gold": os.path.join(BASE_PATH, "DataToValidate", "Triplets-spider.json"),
                        "pred": os.path.join(BASE_PATH, "ExtractedResults2", f"Reconciled_chunk-spider_{file_suffix}")
                    },
                    {
                        "gold": os.path.join(BASE_PATH, "DataToValidate", "Triplets China cyber espionage.json"),
                        "pred": os.path.join(BASE_PATH, "ExtractedResults2", f"Reconciled_chunks China's Cyber Espionage_{file_suffix}")
                    },
                    {
                        "gold": os.path.join(BASE_PATH, "DataToValidate", "Triplets-chunk-m-trends-2025.json"),
                        "pred": os.path.join(BASE_PATH, "ExtractedResults2", f"Reconciled_chunks-m-trends-2025_{file_suffix}")
                    }
                ]

                print(f"\n\n{'#'*80}")
                print(f"🚀 TEST : ROCADE={test_rocade} | CHUNKING={test_chunking} | PROMPT={test_prompt.upper()}")
                print(f"{'#'*80}")

                evaluate_micro_average(
                    evaluation_pairs=EVALUATION_PAIRS,
                    similarity_threshold=SIMILARITY_THRESHOLD
                )