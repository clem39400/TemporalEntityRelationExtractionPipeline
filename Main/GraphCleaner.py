import json
import os

def dynamic_cleaner(input_path, output_path, max_connectivity=15):
    with open(input_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)

    # 1. Calcul de la connectivité (Degré des nœuds)
    degree = {}
    for edge in graph.get("edges", []):
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    ids_to_remove = set()
    for node in graph.get("nodes", []):
        node_id = node["id"]

        # CONDITION A : Trop connecté (Super-nœud polluant type "Windows")
        if degree.get(node_id, 0) > max_connectivity:
            ids_to_remove.add(node_id)
            continue

        # CONDITION B : Nœud orphelin (Degré 0)
        if degree.get(node_id, 0) == 0:
            ids_to_remove.add(node_id)

    # 3. Reconstruction du graphe
    new_nodes = [n for n in graph["nodes"] if n["id"] not in ids_to_remove]
    new_edges = [e for e in graph["edges"] if e["source"] not in ids_to_remove and e["target"] not in ids_to_remove]

    graph["metadata"]["total_nodes"] = len(new_nodes)
    graph["metadata"]["total_relations"] = len(new_edges)
    graph["nodes"] = new_nodes
    graph["edges"] = new_edges

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=4, ensure_ascii=False)

    print(f"[*] Nettoyage dynamique fini : {len(ids_to_remove)} nœuds supprimés (Orphelins ou Super-nœuds).")