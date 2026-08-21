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
        Charge enterprise-attack.json (ou attack_dict.json) pour récupérer les noms exacts des techniques.
        """
        if not os.path.exists(path):
            print(f"[!] Dictionnaire introuvable : {path}")
            return {}

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Gestion du format natif STIX/ATT&CK du MITRE si on pointe vers enterprise-attack.json
        attack_dict = {}
        for obj in data.get("objects", []):
            if obj.get("type") == "attack-pattern":
                # Récupération de l'ID ATT&CK (ex: T1059) depuis les external_references
                external_ids = [ref["external_id"] for ref in obj.get("external_references", []) if ref.get("source_name") == "mitre-attack"]
                if external_ids:
                    attack_dict[external_ids[0]] = obj.get("name", "Unknown")

        return attack_dict

    def extract_tram_ground_truth_from_sqlite(self, db_path: str, default_actor: str = "threat actor", max_sentence_id: int = None) -> dict:
        if not os.path.exists(db_path):
            print(f"[!] Base de données SQLite introuvable : {db_path}")
            return {"nodes": [], "edges": []}

        gt_entities = {}
        gt_relations = []

        actor_id = "G_ACTOR_1"
        gt_entities[actor_id] = {
            "id": actor_id,
            "type": "Threat_Actor",
            "labels": [default_actor]
        }

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            ao_columns = [col[1] for col in cursor.execute("PRAGMA table_info(tram_attackobject);").fetchall()]
            id_col = next((c for c in ao_columns if "external" in c or "stix" in c or "key" in c or "id" in c and c != "id"), "attack_id")
            name_col = next((c for c in ao_columns if "name" in c or "title" in c), "name")

            # Ajout du filtre optionnel sur l'ID de la phrase
            where_clause = f"WHERE s.id <= {max_sentence_id}" if max_sentence_id else ""

            query = f"""
                SELECT s.id, s.text, ao.{id_col}, ao.{name_col} 
                FROM tram_sentence s
                JOIN tram_mapping m ON s.id = m.sentence_id
                JOIN tram_attackobject ao ON m.attack_object_id = ao.id
                {where_clause}
            """

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            tech_counter = 1
            for row in rows:
                sentence_id, sentence_text, attack_id, tech_name_raw = row
                if not attack_id:
                    continue

                tech_name = self.attack_dict.get(attack_id, tech_name_raw or "Unknown_Technique")
                rocade_label = f"{attack_id} - {tech_name}"

                tech_node_id = f"G_TECH_{tech_counter}"

                gt_entities[tech_node_id] = {
                    "id": tech_node_id,
                    "type": "Attack_Pattern",
                    "labels": [rocade_label]
                }

                gt_relations.append({
                    "source": actor_id,
                    "target": tech_node_id,
                    "relation_type": "USES",
                    "provenance": f"tram_sqlite_row_{tech_counter}_sentence_{sentence_id}"
                })
                tech_counter += 1

        except Exception as e:
            print(f"[!] Erreur SQL : {e}")

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
    DB_PATH = r"C:\Users\cleme\IdeaProjects\tram\data\db.sqlite3"
    # Point vers le fichier enterprise-attack.json présent dans ton dossier data/attack de TRAM
    ATTACK_DICT_PATH = r"C:\Users\cleme\IdeaProjects\tram\data\attack\enterprise-attack.json"

    tram_utils = TRAMUtils(ATTACK_DICT_PATH)
    ground_truth_graph = tram_utils.extract_tram_ground_truth_from_sqlite(DB_PATH, default_actor="threat actor")

    print(json.dumps(ground_truth_graph, indent=4, ensure_ascii=False))