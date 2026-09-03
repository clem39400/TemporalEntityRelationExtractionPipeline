import json
import os
from pyvis.network import Network

def visualize_graph(json_path, output_html):
    if not os.path.exists(json_path):
        print("Erreur : Le graphe réconcilié n'existe pas encore.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Initialisation du réseau Pyvis avec un fond sombre moderne
    net = Network(height="850px", width="100%", bgcolor="#1e1e1e", font_color="#ffffff", notebook=False, directed=True)

    # Palette de couleurs cyber modernisée et adoucie
    color_map = {
        "Threat_Actor": "#ff5252",        # Rouge vif
        "Attack_Pattern": "#ffb142",      # Orange ambré
        "Malware": "#ffda79",             # Jaune doux
        "Tool": "#4bcffa",                # Bleu clair
        "Vulnerability": "#cd84f1",       # Violet pastel
        "Attacker_Infrastructure": "#ff793f", # Corail
        "Victim_Asset": "#2ed573",         # Vert émeraude
        "Observable": "#747d8c"           # Gris ardoise
    }

    # Calcul du degré de connexion de chaque nœud pour moduler leur taille
    node_degrees = {}
    for edge in graph_data.get("edges", []):
        node_degrees[edge["source"]] = node_degrees.get(edge["source"], 0) + 1
        node_degrees[edge["target"]] = node_degrees.get(edge["target"], 0) + 1

    # Ajout des nœuds avec mise en forme dynamique
    for node in graph_data["nodes"]:
        node_id = node["id"]
        label = node["labels"][0] if node["labels"] else node_id
        node_type = node["type"]
        color = color_map.get(node_type, "#ffffff")

        # Taille proportionnelle au nombre de liens (min 15, max 35)
        degree = node_degrees.get(node_id, 0)
        size = max(15, min(35, 12 + degree * 2))

        net.add_node(
            node_id,
            label=label,
            title=f"Type: {node_type}\nConnexions (Degré): {degree}\nAliases: {', '.join(node['labels'])}",
            color=color,
            size=size,
            borderWidth=2,
            borderColor="#ffffff"
        )

    # Ajout des relations avec styles différenciés (sémantique vs temporel)
    for edge in graph_data["edges"]:
        is_temporal = edge["relation_type"] in ["BEFORE", "SIMULTANEOUS"]
        label = edge["relation_type"]

        net.add_edge(
            edge["source"],
            edge["target"],
            label=label,
            color="#a4b0be" if is_temporal else "#70a1ff",
            arrows="to",
            dashes=is_temporal, # Pointillé pour le temps, plein pour le sémantique
            width=2,
            font={"size": 10, "color": "#ced6e0", "align": "middle"}
        )

    # Configuration physique avancée (évite les superpositions et stabilise le graphe)
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3500,
          "centralGravity": 0.3,
          "springLength": 100,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.2
        },
        "maxVelocity": 30,
        "solver": "barnesHut",
        "timestep": 0.3
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": true
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "roundness": 0.2
        }
      }
    }
    """)

    net.save_graph(output_html)
    print(f"Visualisation améliorée générée avec succès : {output_html}")