import json
import os
from sentence_transformers import SentenceTransformer, util

# Import de ton nouvel utilitaire TRAM
from Main.TRAMUtils import TRAMUtils

def get_triples_from_graph_dict(graph_data):
    """
    Extrait un set de triplets (Source, Relation, Cible) à partir d'un dictionnaire
    contenant des 'nodes' et des 'edges' (Format ROCADE réconcilié).
    """
    # On mappe l'ID du nœud vers son label textuel principal (en minuscules pour l'évaluation)
    node_map = {node["id"]: node["labels"][0].lower().strip() for node in graph_data.get("nodes", [])}
    triples = set()

    for edge in graph_data.get("edges", []):
        src_label = node_map.get(edge["source"], "")
        tgt_label = node_map.get(edge["target"], "")
        rel_type = edge["relation_type"].upper()

        # On n'ajoute que si la source et la cible ont bien été trouvées
        if src_label and tgt_label:
            triples.add((src_label, rel_type, tgt_label))

    return triples

def load_predicted_triples(predicted_graph_path):
    """Charge les prédictions générées par le LLM (Graphe réconcilié/nettoyé)."""
    with open(predicted_graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return get_triples_from_graph_dict(graph_data)

def load_tram_ground_truth(tram_path, attack_dict_path, default_actor="threat actor"):
    """Charge la vérité terrain depuis TRAM et la convertit en triplets ROCADE."""
    tram_utils = TRAMUtils(attack_dict_path)
    gt_graph_data = tram_utils.extract_tram_ground_truth(tram_path, default_actor=default_actor)
    return get_triples_from_graph_dict(gt_graph_data)

def evaluate_pipeline(tram_export_path, attack_dict_path, predicted_graph_path, default_actor="threat actor", similarity_threshold=0.80):
    print(f"\n{'='*50}")
    print("ÉVALUATION DES PERFORMANCES DU GRAPHE (TRAM)")
    print(f"{'='*50}")

    # Chargement unifié via la nouvelle structure
    gt_triples = load_tram_ground_truth(tram_export_path, attack_dict_path, default_actor)
    pred_triples = load_predicted_triples(predicted_graph_path)

    print(f"[*] Vérité terrain TRAM (Total) : {len(gt_triples)} triplets")
    print(f"[*] Prédictions LLM (Total)     : {len(pred_triples)} triplets")

    if len(gt_triples) == 0:
        print("\n[!] Aucune vérité terrain trouvée. Impossible de calculer les métriques.")
        return

    # --- 1. EXACT MATCH ---
    exact_tp_set = gt_triples.intersection(pred_triples)
    exact_tp = len(exact_tp_set)

    remaining_pred = list(pred_triples - exact_tp_set)
    remaining_gt = list(gt_triples - exact_tp_set)

    # --- 2. SOFT MATCH ---
    print("\n[*] Chargement du modèle de similarité sémantique pour le Soft Match...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

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
            matched_pred_indices.add(p_idx) # On mémorise la prédiction repêchée

    # --- 3. ISOLER LES ERREURS POUR L'AFFICHAGE ---
    final_fp_list = [pred for idx, pred in enumerate(remaining_pred) if idx not in matched_pred_indices]
    final_fn_list = [gt for idx, gt in enumerate(remaining_gt) if idx not in matched_gt_indices]

    # --- 4. CALCUL DES MÉTRIQUES ---
    exact_fp = len(pred_triples) - exact_tp
    exact_fn = len(gt_triples) - exact_tp
    e_prec = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
    e_rec = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
    e_f1 = 2 * (e_prec * e_rec) / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0

    total_tp = exact_tp + soft_tp
    soft_fp = len(pred_triples) - total_tp
    soft_fn = len(gt_triples) - total_tp
    s_prec = total_tp / (total_tp + soft_fp) if (total_tp + soft_fp) > 0 else 0.0
    s_rec = total_tp / (total_tp + soft_fn) if (total_tp + soft_fn) > 0 else 0.0
    s_f1 = 2 * (s_prec * s_rec) / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    # --- 5. AFFICHAGE DES RÉSULTATS ---
    print(f"\n[MÉTRIQUES EXACT MATCH]")
    print(f"- Vrais Positifs (TP)  : {exact_tp}")
    print(f"- Faux Positifs (FP)   : {exact_fp}")
    print(f"- Faux Négatifs (FN)   : {exact_fn}")
    print("-" * 30)
    print(f"Précision : {e_prec:.4f} | Rappel : {e_rec:.4f} | F1-Score : {e_f1:.4f}")

    print(f"\n[MÉTRIQUES SOFT MATCH (Seuil de tolérance: {similarity_threshold})]")
    print(f"- Vrais Positifs (TP)  : {total_tp} (dont {soft_tp} repêchés par similarité)")
    print(f"- Faux Positifs (FP)   : {soft_fp}")
    print(f"- Faux Négatifs (FN)   : {soft_fn}")
    print("-" * 30)
    print(f"Précision : {s_prec:.4f} | Rappel : {s_rec:.4f} | F1-Score : {s_f1:.4f}\n")

    # --- 6. DEBUG: AFFICHAGE DES ERREURS ---
    print(f"{'='*50}")
    print("ANALYSE DES ERREURS (Après Soft Match)")
    print(f"{'='*50}")

    print("\n[FAUX POSITIFS] - Générés par le LLM mais absents ou rejetés par TRAM :")
    if not final_fp_list:
        print("  Aucun.")
    for src, rel, tgt in sorted(final_fp_list):
        print(f"  (+) '{src}' -> [{rel}] -> '{tgt}'")

    print("\n[FAUX NÉGATIFS] - Présents dans TRAM mais manqués par le LLM :")
    if not final_fn_list:
        print("  Aucun.")
    for src, rel, tgt in sorted(final_fn_list):
        print(f"  (-) '{src}' -> [{rel}] -> '{tgt}'")
    print(f"{'='*50}\n")

    return e_f1, s_f1

if __name__ == "__main__":
    # Chemins à mettre à jour selon ton arborescence locale
    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"

    # Nouveaux chemins pour pointer vers les données TRAM
    TRAM_EXPORT_PATH = os.path.join(BASE_PATH, "Data", "tram_export.json")
    ATTACK_DICT_PATH = os.path.join(BASE_PATH, "Data", "attack_dict.json")

    # Ton graphe généré par la pipeline
    PREDICTED_GRAPH_PATH = os.path.join(BASE_PATH, "ExtractedResults", "Cleaned_few_shot.json")

    # Le default_actor permet de donner un sujet (ex: "APT29", "threat actor") pour lier les techniques TRAM
    evaluate_pipeline(
        tram_export_path=TRAM_EXPORT_PATH,
        attack_dict_path=ATTACK_DICT_PATH,
        predicted_graph_path=PREDICTED_GRAPH_PATH,
        default_actor="threat actor",
        similarity_threshold=0.80
    )