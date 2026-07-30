import json

def extract_casie_ground_truth(casie_file_path):
    """
    Parse un fichier d'annotation CASIE et extrait les entités et les relations
    sous forme de triplets (Source, Prédicat, Objet) pour l'évaluation.
    """
    with open(casie_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 1. Cartographie des entités (Nœuds)
    # Permet de retrouver le texte et le type à partir de l'ID CASIE (ex: "T1")
    entities_map = {}
    ground_truth_nodes = []

    for entity in data.get("cybNER", []):
        entities_map[entity["id"]] = {
            "text": entity["text"],
            "type": entity["type"]
        }
        ground_truth_nodes.append({
            "text": entity["text"],
            "type": entity["type"]
        })

    # 2. Dictionnaire de traduction (Mapping CASIE -> ROCADE)
    # À enrichir selon l'exhaustivité des événements CASIE rencontrés
    event_mapping = {
        "Vulnerability_Exploit": "EXPLOITS",
        "Malware_Deployment": "USES",
        "Attack_Pattern": "USES",
        "Target_Compromise": "TARGETS"
    }

    ground_truth_edges = []

    # 3. Extraction des triplets (Arêtes)
    for event in data.get("events", []):
        casie_event_type = event.get("type")
        relation_type = event_mapping.get(casie_event_type, "RELATED_TO")

        arguments = event.get("arguments", [])

        source_entity = None
        target_entity = None

        # Identification du sens de la relation basée sur les rôles sémantiques
        for arg in arguments:
            role = arg.get("role")
            entity_id = arg.get("entity_id")

            # Définition heuristique : l'attaquant ou l'outil est généralement la source
            if role in ["Attacker", "Tool", "Malware_Source"]:
                source_entity = entities_map.get(entity_id)
            # La vulnérabilité, le malware déployé ou la victime est la cible
            elif role in ["Vulnerability", "Malware", "Victim", "Compromised_System"]:
                target_entity = entities_map.get(entity_id)

        # Construction du triplet final si les deux extrémités sont trouvées
        if source_entity and target_entity:
            ground_truth_edges.append({
                "source": source_entity["text"],
                "source_type": source_entity["type"],
                "relation_type": relation_type,
                "target": target_entity["text"],
                "target_type": target_entity["type"]
            })

    return {
        "text": data.get("text", ""),
        "nodes": ground_truth_nodes,
        "edges": ground_truth_edges
    }
