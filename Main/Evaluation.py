import json
import os
from sentence_transformers import SentenceTransformer, util

# Import de ton utilitaire TRAM
from Main.TRAMUtils import TRAMUtils

def get_triples_from_graph_dict(graph_data):
    """
    Extrait un set de triplets (Source, Relation, Cible) à partir d'un dictionnaire
    contenant des 'nodes' et des 'edges' (Format ROCADE réconcilié).
    """
    node_map = {node["id"]: node["labels"][0].lower().strip() for node in graph_data.get("nodes", [])}
    triples = set()

    for edge in graph_data.get("edges", []):
        src_label = node_map.get(edge["source"], "")
        tgt_label = node_map.get(edge["target"], "")
        rel_type = edge["relation_type"].upper()

        if src_label and tgt_label:
            triples.add((src_label, rel_type, tgt_label))

    return triples

def load_predicted_triples_with_normalization(predicted_graph_path, attack_dict_path):
    """
    Charge les prédictions du LLM et tente de mapper les libellés textuels
    vers les IDs ATT&CK officiels en utilisant le dictionnaire.
    """
    with open(predicted_graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Charger le dictionnaire ATT&CK pour faire la correspondance inverse (Nom -> ID)
    with open(attack_dict_path, 'r', encoding='utf-8') as f:
        attack_data = json.load(f)

    name_to_id = {}
    for obj in attack_data.get("objects", []):
        if obj.get("type") == "attack-pattern":
            name = obj.get("name", "").lower().strip()
            external_ids = [ref["external_id"] for ref in obj.get("external_references", []) if ref.get("source_name") == "mitre-attack"]
            if external_ids and name:
                name_to_id[name] = external_ids[0].lower()

    node_map = {node["id"]: node["labels"][0].lower().strip() for node in graph_data.get("nodes", [])}
    triples = set()

    for edge in graph_data.get("edges", []):
        src_label = node_map.get(edge["source"], "").replace("attackers", "threat actor") # Normalisation de l'acteur
        tgt_label = node_map.get(edge["target"], "")
        rel_type = edge["relation_type"].upper()

        if src_label and tgt_label:
            # Tentative de correspondance avec un ID ATT&CK si le texte y ressemble
            normalized_tgt = tgt_label
            for tech_name, tech_id in name_to_id.items():
                if tech_name in tgt_label or tgt_label in tech_name:
                    normalized_tgt = f"{tech_id} - {tech_name}"
                    break

            triples.add((src_label, rel_type, normalized_tgt))

    return triples

def load_tram_ground_truth_from_db(tram_db_path, attack_dict_path, default_actor="threat actor", max_sentence_id=None):
    """Charge la vérité terrain directement depuis la base SQLite TRAM et la convertit en triplets ROCADE."""
    tram_utils = TRAMUtils(attack_dict_path)
    # Utilisation de la méthode SQLite avec la limite de phrases
    gt_graph_data = tram_utils.extract_tram_ground_truth_from_sqlite(tram_db_path, default_actor=default_actor, max_sentence_id=max_sentence_id)
    return get_triples_from_graph_dict(gt_graph_data)

def evaluate_pipeline(tram_db_path, attack_dict_path, predicted_graph_path, default_actor="threat actor", similarity_threshold=0.80, max_sentence_id=None):
    print(f"\n{'='*50}")
    print("ÉVALUATION DES PERFORMANCES DU GRAPHE (TRAM SQLite)")
    print(f"{'='*50}")

    # Chargement unifié depuis la base SQLite TRAM avec la limite activée
    gt_triples = load_tram_ground_truth_from_db(tram_db_path, attack_dict_path, default_actor, max_sentence_id)
    pred_triples = load_predicted_triples_with_normalization(predicted_graph_path, attack_dict_path)


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
            matched_pred_indices.add(p_idx)

    print("\n--- EXEMPLE DE VÉRITÉ TERRAIN (TRAM) ---")
    print(list(gt_triples)[:3])

    print("\n--- EXEMPLE DE PRÉDICTIONS (LLM) ---")
    print(list(pred_triples)[:3])

    # --- 3. CALCUL DES MÉTRIQUES ---
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

    # --- NOUVELLE ÉVALUATION : EXTRACTION DES TECHNIQUES (ENTITÉS) ---
    print(f"\n{'='*50}")
    print("ÉVALUATION CIBLÉE : EXTRACTION DES TECHNIQUES MITRE ATT&CK")
    print(f"{'='*50}")

    # On isole uniquement les cibles (targets) qui commencent par "t" suivi d'un chiffre (les IDs ATT&CK)
    gt_techniques = set([tgt for _, _, tgt in gt_triples if tgt.startswith('t1')])
    pred_techniques = set([tgt for _, _, tgt in pred_triples if tgt.startswith('t1')])

    tech_tp = len(gt_techniques.intersection(pred_techniques))
    tech_fp = len(pred_techniques - gt_techniques)
    tech_fn = len(gt_techniques - pred_techniques)

    tech_prec = tech_tp / (tech_tp + tech_fp) if (tech_tp + tech_fp) > 0 else 0.0
    tech_rec = tech_tp / (tech_tp + tech_fn) if (tech_tp + tech_fn) > 0 else 0.0
    tech_f1 = 2 * (tech_prec * tech_rec) / (tech_prec + tech_rec) if (tech_prec + tech_rec) > 0 else 0.0

    print(f"- Techniques Vraies Positives (TP) : {tech_tp}")
    print(f"- Techniques Fausses Positives (FP) : {tech_fp}")
    print(f"- Techniques Fausses Négatives (FN) : {tech_fn}")
    print("-" * 30)
    print(f"Précision: {tech_prec:.4f} | Rappel: {tech_rec:.4f} | F1-Score: {tech_f1:.4f}\n")

    # --- 4. AFFICHAGE DES RÉSULTATS ---
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

    return e_f1, s_f1

if __name__ == "__main__":
    # Chemins mis à jour pour pointer vers ta base SQLite TRAM et ton dictionnaire STIX
    TRAM_DB_PATH = r"C:\Users\cleme\IdeaProjects\tram\data\db.sqlite3"
    ATTACK_DICT_PATH = r"C:\Users\cleme\IdeaProjects\tram\data\attack\enterprise-attack.json"

    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"
    PREDICTED_GRAPH_PATH = os.path.join(BASE_PATH, "ExtractedResults", "Cleaned_few_shot.json")

    evaluate_pipeline(
        tram_db_path=TRAM_DB_PATH,
        attack_dict_path=ATTACK_DICT_PATH,
        predicted_graph_path=PREDICTED_GRAPH_PATH,
        default_actor="threat actor",
        similarity_threshold=0.45
    )