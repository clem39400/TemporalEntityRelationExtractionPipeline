import json
import os
import re
from sentence_transformers import SentenceTransformer, util


# --- NOUVEAU : Règles strictes ---
# Expressions régulières pour cibler les identifiants normés (CVE, UNC, APT, FIN)
REGEX_STRICT_ENTITIES = [
    re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE),
    re.compile(r"^(UNC|APT|FIN)\d+$", re.IGNORECASE)
]

def requires_strict_match(mention: str) -> bool:
    """Vérifie si la mention correspond à un identifiant strict qui ne doit pas être fusionné par similarité."""
    mention_clean = mention.strip()
    return any(regex.search(mention_clean) for regex in REGEX_STRICT_ENTITIES)

# NOUVEAU : Ajout du paramètre malware_threshold
def reconcile_graph(input_path, output_path, similarity_threshold=0.85, malware_threshold=0.95):
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} est introuvable.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)

    print("Chargement du modèle de similarité sémantique...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    global_entities = {} # {global_id: {type, mentions, embedding, is_strict}}
    global_relations = []
    local_to_global_map = {} # { "C331_E1": "G_E1" }
    global_id_counter = 1

    print(f"Analyse de {len(extracted_data)} extractions locales...")

    for entry in extracted_data:
        if not entry.get("extraction"): continue

        chunk_id = entry["chunk_metadata"]["chunk_id"]
        entities = entry["extraction"].get("entities", [])
        relations = entry["extraction"].get("relations", [])

        # 1. Résolution d'Entités (Entity Resolution)
        for ent in entities:
            local_id = ent["id"]
            ent_type = ent["type"]
            mention = ent["mention"].strip()
            mention_lower = mention.lower()

            # Déterminer si l'entité courante est stricte
            is_strict_local = requires_strict_match(mention)

            ent_emb = model.encode(mention_lower, convert_to_tensor=True)
            matched_global_id = None

            # Comparaison avec les entités globales existantes
            for g_id, g_data in global_entities.items():
                if g_data["type"] == ent_type:
                    is_match = False

                    # --- NOUVEAU : Logique de comparaison hybride ---
                    # Si l'entité locale ou l'entité globale est "stricte" (ex: CVE-2024-3400)
                    if is_strict_local or g_data["is_strict"]:
                        # On exige une correspondance lexicale parfaite
                        if mention_lower in g_data["mentions_lower"]:
                            is_match = True
                    else:
                        # Sinon, on utilise la similarité sémantique
                        sim = util.cos_sim(ent_emb, g_data["embedding"]).item()

                        # Seuil dynamique : plus haut si c'est un Malware
                        current_threshold = malware_threshold if ent_type == "Malware" else similarity_threshold

                        if mention_lower in g_data["mentions_lower"] or sim >= current_threshold:
                            is_match = True

                    if is_match:
                        matched_global_id = g_id
                        g_data["mentions"].add(mention)
                        g_data["mentions_lower"].add(mention_lower)
                        break

            if matched_global_id:
                local_to_global_map[local_id] = matched_global_id
            else:
                new_g_id = f"G_E{global_id_counter}"
                global_entities[new_g_id] = {
                    "type": ent_type,
                    "mentions": {mention},
                    "mentions_lower": {mention_lower},
                    "embedding": ent_emb,
                    "is_strict": is_strict_local # NOUVEAU : Sauvegarde du statut strict
                }
                local_to_global_map[local_id] = new_g_id
                global_id_counter += 1

        # 2. Migration des Relations (Relation Mapping)
        for rel in relations:
            src_global = local_to_global_map.get(rel["source"])
            tgt_global = local_to_global_map.get(rel["target"])

            if src_global and tgt_global:
                global_relations.append({
                    "source": src_global,
                    "target": tgt_global,
                    "relation_type": rel["relation_type"],
                    "provenance": f"chunk_{chunk_id}"
                })

    # --- FORMATAGE FINAL ---
    final_nodes = [
        {"id": k, "type": v["type"], "labels": list(v["mentions"])}
        for k, v in global_entities.items()
    ]

    final_graph = {
        "metadata": {"total_nodes": len(final_nodes), "total_relations": len(global_relations)},
        "nodes": final_nodes,
        "edges": global_relations
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_graph, f, indent=4, ensure_ascii=False)

    print(f"Réconciliation terminée : {len(final_nodes)} entités globales créées.")

if __name__ == "__main__":

    '''
    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"
    INPUT_RESULTS = os.path.join(BASE_PATH, "ExtractedResults", "results_few_shot.json")
    OUTPUT_GRAPH = os.path.join(BASE_PATH, "ExtractedResults", "Reconciled_Knowledge_Graph.json")
    reconcile_graph(INPUT_RESULTS, OUTPUT_GRAPH)
    '''