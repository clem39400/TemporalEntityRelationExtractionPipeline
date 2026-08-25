import json
import os
from sentence_transformers import SentenceTransformer, util

# Import de ton utilitaire TRAM
from Main.TRAMUtils import TRAMUtils

def get_triples_from_graph_dict(graph_data, relation_filter=None):
    """
    Extrait un set de triplets (Source, Relation, Cible) à partir d'un dictionnaire.
    Ajout d'un paramètre 'relation_filter' pour isoler les liens temporels.
    """
    node_map = {node["id"]: node["labels"][0].lower().strip() for node in graph_data.get("nodes", [])}
    triples = set()

    for edge in graph_data.get("edges", []):
        src_label = node_map.get(edge.get("source"), "")
        tgt_label = node_map.get(edge.get("target"), "")
        rel_type = edge.get("relation_type", "").upper()

        # Filtre pour l'évaluation spécifique (ex: uniquement "BEFORE")
        if relation_filter and rel_type != relation_filter.upper():
            continue

        if src_label and tgt_label:
            triples.add((src_label, rel_type, tgt_label))

    return triples

def calculate_metrics(title, gt_triples, pred_triples, model, similarity_threshold=0.80):
    """
    Fonction centralisée pour calculer et afficher l'Exact Match et le Soft Match.
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"[*] Vérité terrain (Total) : {len(gt_triples)} triplets")
    print(f"[*] Prédictions LLM (Total) : {len(pred_triples)} triplets")

    if len(gt_triples) == 0:
        print("\n[!] Aucune vérité terrain trouvée pour cette catégorie.")
        return

    # --- 1. EXACT MATCH ---
    exact_tp_set = gt_triples.intersection(pred_triples)
    exact_tp = len(exact_tp_set)
    remaining_pred = list(pred_triples - exact_tp_set)
    remaining_gt = list(gt_triples - exact_tp_set)

    # --- 2. SOFT MATCH ---
    emb_cache = {}
    def get_embedding(text):
        if text not in emb_cache:
            emb_cache[text] = model.encode(text, convert_to_tensor=True)
        return emb_cache[text]

    soft_tp = 0
    matched_gt_indices = set()
    matched_pred_indices = set()

    for p_idx, (p_src, p_rel, p_tgt) in enumerate(remaining_pred):
        best_sim = 0
        best_gt_idx = -1

        p_src_emb = get_embedding(p_src)
        p_tgt_emb = get_embedding(p_tgt)

        for g_idx, (g_src, g_rel, g_tgt) in enumerate(remaining_gt):
            if g_idx in matched_gt_indices:
                continue

            if p_rel != g_rel:
                continue

            g_src_emb = get_embedding(g_src)
            g_tgt_emb = get_embedding(g_tgt)

            sim_src = util.cos_sim(p_src_emb, g_src_emb).item()
            sim_tgt = util.cos_sim(p_tgt_emb, g_tgt_emb).item()

            if sim_src >= similarity_threshold and sim_tgt >= similarity_threshold:
                avg_sim = (sim_src + sim_tgt) / 2
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_gt_idx = g_idx

        if best_gt_idx != -1:
            soft_tp += 1
            matched_gt_indices.add(best_gt_idx)
            matched_pred_indices.add(p_idx)

    # --- 3. CALCUL DES MÉTRIQUES ---
    total_tp = exact_tp + soft_tp
    soft_fp = len(pred_triples) - total_tp
    soft_fn = len(gt_triples) - total_tp

    s_prec = total_tp / (total_tp + soft_fp) if (total_tp + soft_fp) > 0 else 0.0
    s_rec = total_tp / (total_tp + soft_fn) if (total_tp + soft_fn) > 0 else 0.0
    s_f1 = 2 * (s_prec * s_rec) / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    print(f"\n[MÉTRIQUES SOFT MATCH (Seuil de tolérance: {similarity_threshold})]")
    print(f"- Vrais Positifs (TP)  : {total_tp} (Exact: {exact_tp}, Soft: {soft_tp})")
    print(f"- Faux Positifs (FP)   : {soft_fp}")
    print(f"- Faux Négatifs (FN)   : {soft_fn}")
    print("-" * 30)
    print(f"Précision : {s_prec:.4f} | Rappel : {s_rec:.4f} | F1-Score : {s_f1:.4f}\n")


def evaluate_custom_gold_standard(gold_json_path, pred_json_path, similarity_threshold=0.80):
    """
    Point d'entrée pour évaluer le Gold Standard manuel (comparaison de deux JSON ROCADE purs).
    """
    if not os.path.exists(gold_json_path) or not os.path.exists(pred_json_path):
        print(f"[!] Fichiers introuvables. Vérifiez les chemins:\n- {gold_json_path}\n- {pred_json_path}")
        return

    with open(gold_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    print("\n[*] Chargement du modèle de similarité sémantique (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # ÉVALUATION 1 : GLOBALE (Extraction Entités et Sémantique)
    gt_all = get_triples_from_graph_dict(gt_data)
    pred_all = get_triples_from_graph_dict(pred_data)
    calculate_metrics("1. ÉVALUATION GLOBALE (TOUTES RELATIONS ROCADE)", gt_all, pred_all, model, similarity_threshold)

    # ÉVALUATION 2 : TEMPORELLE (Cohérence chronologique)
    gt_temporal = get_triples_from_graph_dict(gt_data, relation_filter="BEFORE")
    pred_temporal = get_triples_from_graph_dict(pred_data, relation_filter="BEFORE")
    calculate_metrics("2. ÉVALUATION TEMPORELLE (CHRONOLOGIE 'BEFORE')", gt_temporal, pred_temporal, model, similarity_threshold)


if __name__ == "__main__":
    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"

    # 1. Chemin vers le Gold Standard manuel que nous venons de créer
    GOLD_STANDARD_PATH = os.path.join(BASE_PATH, "DataToValidate", "Triplets-chunk-m-trends-2025.json")

    # 2. Chemin vers les résultats générés par ton script Main.py (Phase 3)
    # Assure-toi que ce nom correspond bien au fichier que ta pipeline génère actuellement
    PREDICTED_GRAPH_PATH = os.path.join(BASE_PATH, "ExtractedResults2", "Reconciled_cot.json")

    evaluate_custom_gold_standard(
        gold_json_path=GOLD_STANDARD_PATH,
        pred_json_path=PREDICTED_GRAPH_PATH,
        similarity_threshold=0.5
    )