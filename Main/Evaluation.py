import json
import os
from sentence_transformers import SentenceTransformer, util

# Import de ton utilitaire TRAM
from Main.TRAMUtils import TRAMUtils

def get_triples_from_graph_dict(graph_data, relation_filter=None):
    """
    Extrait un set de triplets (Source, Relation, Cible) à partir d'un dictionnaire[cite: 30].
    """
    node_map = {node["id"]: node["labels"][0].lower().strip() for node in graph_data.get("nodes", [])}
    triples = set()

    for edge in graph_data.get("edges", []):
        src_label = node_map.get(edge.get("source"), "")
        tgt_label = node_map.get(edge.get("target"), "")
        rel_type = edge.get("relation_type", "").upper()

        if relation_filter and rel_type != relation_filter.upper():
            continue

        if src_label and tgt_label:
            triples.add((src_label, rel_type, tgt_label))

    return triples

def calculate_independent_metrics(gt_triples, pred_triples, model, similarity_threshold=0.80):
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


def evaluate_micro_average(evaluation_pairs: list, similarity_threshold=0.80):
    """
    Évalue indépendamment chaque rapport puis calcule la moyenne (Micro-Average).
    """
    print("\n[*] Chargement du modèle de similarité sémantique (all-MiniLM-L6-v2)...")
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

        with open(gold_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        # Évaluation Globale pour ce rapport
        gt_all = get_triples_from_graph_dict(gt_data)
        pred_all = get_triples_from_graph_dict(pred_data)
        tp, fp, fn = calculate_independent_metrics(gt_all, pred_all, model, similarity_threshold)
        global_tp += tp; global_fp += fp; global_fn += fn

        # Évaluation Temporelle pour ce rapport
        gt_temp = get_triples_from_graph_dict(gt_data, relation_filter="BEFORE")
        pred_temp = get_triples_from_graph_dict(pred_data, relation_filter="BEFORE")
        t_tp, t_fp, t_fn = calculate_independent_metrics(gt_temp, pred_temp, model, similarity_threshold)
        temp_tp += t_tp; temp_fp += t_fp; temp_fn += t_fn

    # Affichage des résultats consolidés
    print_micro_average_metrics("1. ÉVALUATION GLOBALE (MICRO-AVERAGE)", global_tp, global_fp, global_fn, similarity_threshold)
    print_micro_average_metrics("2. ÉVALUATION TEMPORELLE (MICRO-AVERAGE 'BEFORE')", temp_tp, temp_fp, temp_fn, similarity_threshold)


if __name__ == "__main__":
    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"

    EVALUATION_PAIRS = [
        {
            "gold": os.path.join(BASE_PATH, "DataToValidate", "Triplets-spider.json"),
            "pred": os.path.join(BASE_PATH, "ExtractedResults2", "Reconciled_chunk-spider.json")
        }
    ]

    evaluate_micro_average(
        evaluation_pairs=EVALUATION_PAIRS,
        similarity_threshold=0.5
    )