import json
import os
from pyvis.network import Network

# --- CONFIGURATION ---
BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"
INPUT_GRAPH = os.path.join(BASE_PATH, "ExtractedResults", "Cleaned_Knowledge_Graph.json")
OUTPUT_HTML = os.path.join(BASE_PATH, "ExtractedResults", "Visualisation_Graphe.html")

def visualize_graph(json_path, output_html):
    if not os.path.exists(json_path):
        print("Erreur : Le graphe réconcilié n'existe pas encore.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Initialisation du réseau Pyvis
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=True)

    # Mapping des couleurs par type d'entité pour la lisibilité
    color_map = {
        "Threat_Actor": "#FF4B4B",        # Rouge
        "Attack_Pattern": "#FFA500",      # Orange
        "Malware": "#FFD700",             # Jaune
        "Tool": "#4B9AFF",                # Bleu
        "Vulnerability": "#9B59B6",       # Violet
        "Attacker_Infrastructure": "#E74C3C", # Corail
        "Victim_Asset": "#2ECC71",         # Vert
        "Observable": "#95A5A6"           # Gris
    }

    # Ajout des nœuds
    for node in graph_data["nodes"]:
        label = node["labels"][0] # On prend la première mention comme label principal
        node_type = node["type"]
        color = color_map.get(node_type, "#FFFFFF")

        net.add_node(node["id"], label=label, title=f"Type: {node_type}\nAliases: {', '.join(node['labels'])}", color=color)

    # Ajout des relations
    for edge in graph_data["edges"]:
        # On distingue les relations temporelles des sémantiques par le style de trait
        is_temporal = edge["relation_type"] in ["BEFORE", "SIMULTANEOUS"]
        label = edge["relation_type"]

        net.add_edge(
            edge["source"],
            edge["target"],
            label=label,
            color="#AAAAAA" if is_temporal else "#5DADE2",
            arrows="to",
            dashes=is_temporal # Trait pointillé pour le temps, plein pour le sémantique
        )

    # Options de physique pour un rendu propre
    net.toggle_physics(True)
    net.save_graph(output_html)
    print(f"Visualisation générée avec succès : {output_html}")

if __name__ == "__main__":
    visualize_graph(INPUT_GRAPH, OUTPUT_HTML)