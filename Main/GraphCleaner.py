import json
import os

# --- CONFIGURATION ---
BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"
INPUT_GRAPH = os.path.join(BASE_PATH, "ExtractedResults", "Reconciled_Knowledge_Graph.json")
OUTPUT_CLEAN_GRAPH = os.path.join(BASE_PATH, "ExtractedResults", "Cleaned_Knowledge_Graph.json")

def dynamic_cleaner(input_path, output_path, max_connectivity=15):
    with open(input_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)

    # 1. Calcul de la connectivité (Degré des nœuds)
    degree = {}
    for edge in graph["edges"]:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    # 2. Identification dynamique des nœuds à supprimer
    ids_to_remove = set()
    for node in graph["nodes"]:
        label = node["labels"][0].lower()
        node_id = node["id"]

        # CONDITION A : Trop connecté (Super-nœud polluant)
        if degree.get(node_id, 0) > max_connectivity:
            ids_to_remove.add(node_id)
            continue

        # CONDITION B : Termes génériques connus (petite liste de base indispensable)
        stop_words = {"threat actor", "attacker", "target", "activity", "adversary"}
        if label in stop_words:
            ids_to_remove.add(node_id)
            continue

        # CONDITION C : Trop court ou sans valeur (ex: "it", "they")
        if len(label) < 3:
            ids_to_remove.add(node_id)

    # 3. Reconstruction du graphe
    new_nodes = [n for n in graph["nodes"] if n["id"] not in ids_to_remove]
    new_edges = [e for e in graph["edges"] if e["source"] not in ids_to_remove and e["target"] not in ids_to_remove]

    # Sauvegarde
    graph["nodes"] = new_nodes
    graph["edges"] = new_edges
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=4, ensure_ascii=False)

    print(f"Nettoyage dynamique fini : {len(ids_to_remove)} nœuds supprimés par analyse statistique.")

if __name__ == "__main__":
    dynamic_cleaner(INPUT_GRAPH, OUTPUT_CLEAN_GRAPH)