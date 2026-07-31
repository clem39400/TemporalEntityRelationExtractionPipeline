import json
import os
import sqlite3

class TRAMUtils:
    def __init__(self, attack_dict_path: str):
        """
        Initialise l'utilitaire avec le dictionnaire officiel du MITRE ATT&CK.
        """
        self.attack_dict = self._load_attack_dict(attack_dict_path)

    def _load_attack_dict(self, path: str) -> dict:
        """
        Charge attack_dict.json pour récupérer les noms exacts des techniques.
        """
        if not os.path.exists(path):
            print(f"[!] Dictionnaire introuvable : {path}")
            return {}

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {tid: details.get("name", "Unknown") for tid, details in data.items()}

    def extract_tram_ground_truth_from_sqlite(self, db_path: str, default_actor: str = "threat actor") -> dict:
        """
        Extrait directement les données de la base SQLite de TRAM pour générer
        le graphe Ground Truth (Vérité Terrain) compatible avec l'ontologie ROCADE.
        """
        if not os.path.exists(db_path):
            print(f"[!] Base de données SQLite introuvable : {db_path}")
            return {"nodes": [], "edges": []}

        gt_entities = {}
        gt_relations = []

        # 1. Création de l'entité source par défaut (ex: "threat actor")
        actor_id = "G_ACTOR_1"
        gt_entities[actor_id] = {
            "id": actor_id,
            "type": "Threat_Actor",
            "labels": [default_actor]
        }

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Requête pour lier les phrases et leurs mappings ATT&CK dans TRAM
            # (Adapte les noms de tables si le schéma local diffère légèrement)
            query = """
                    SELECT s.text, m.attack_id, m.name
                    FROM sentences s
                             JOIN mappings m ON s.id = m.sentence_id \
                    """

            # Si les tables ont une structure différente, on sécurise via un try/except de requête
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            tech_counter = 1
            for row in rows:
                sentence_text, attack_id, tech_name_raw = row
                if not attack_id:
                    continue

                # Résolution du nom via le dictionnaire global si possible
                tech_name = self.attack_dict.get(attack_id, tech_name_raw or "Unknown_Technique")
                rocade_label = f"{attack_id} - {tech_name}"

                tech_node_id = f"G_TECH_{tech_counter}"

                # Ajout du nœud de la technique
                gt_entities[tech_node_id] = {
                    "id": tech_node_id,
                    "type": "Attack_Pattern",
                    "labels": [rocade_label]
                }

                # Création du triplet / relation sémantique (Actor -> USES -> Technique)
                gt_relations.append({
                    "source": actor_id,
                    "target": tech_node_id,
                    "relation_type": "USES",
                    "provenance": f"tram_sqlite_row_{tech_counter}"
                })
                tech_counter += 1

        except Exception as e:
            print(f"[!] Erreur lors de la lecture de la base SQLite TRAM : {e}")

        return {
            "metadata": {
                "total_nodes": len(gt_entities),
                "total_relations": len(gt_relations)
            },
            "nodes": list(gt_entities.values()),
            "edges": gt_relations
        }

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"
    DB_PATH = os.path.join(BASE_PATH, "Data", "tram.db")
    ATTACK_DICT_PATH = os.path.join(BASE_PATH, "Data", "attack_dict.json")

    tram_utils = TRAMUtils(ATTACK_DICT_PATH)
    ground_truth_graph = tram_utils.extract_tram_ground_truth_from_sqlite(DB_PATH, default_actor="threat actor")

    print(json.dumps(ground_truth_graph, indent=4, ensure_ascii=False))