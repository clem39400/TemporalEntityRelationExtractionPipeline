"""
LLMEngine.py
------------
Module chargé de l'inférence LLM pour l'extraction des entités et relations.
S'exécute de manière asynchrone via le ThreadPoolExecutor de l'Orchestrateur.
"""

import json
import os
import logging
import requests

log = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, output_dir: str, rocade_json_path: str, model_name: str = "llama3.1"):
        # CORRECTION 1 : Indentation corrigée
        self.output_dir = output_dir
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. On charge et on distille l'ontologie au démarrage
        log.info(f"[LLM] Chargement de l'ontologie depuis {rocade_json_path}...")
        # CORRECTION 2 : Appel de la méthode avec "self."
        self.rocade_context = self.load_rocade_schema(rocade_json_path)

    def call_llm(self, chunk_text: str, prompt: str) -> dict:
        """
        Appel à l'API locale d'Ollama.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,  # On attend la réponse complète d'un coup
            "format": "json"  # Force Ollama à renvoyer un JSON valide (très utile)
        }

        try:
            # Timeout fixé à 120s pour laisser le temps au LLM local de générer le texte
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()  # Vérifie si la requête a réussi (code 200)

            result = response.json()
            generated_text = result.get("response", "")

            # Parsing basique du JSON renvoyé par le LLM
            try:
                parsed_json = json.loads(generated_text)
            except json.JSONDecodeError:
                # Si le LLM n'a pas renvoyé un JSON parfait
                parsed_json = {"entities": [], "relations": [], "temporal_dependencies": [], "raw_response": generated_text}

            return parsed_json

        except requests.exceptions.RequestException as e:
            log.error(f"[LLM] Erreur lors de l'appel à Ollama : {e}")
            return {
                "entities": [],
                "relations": [],
                "temporal_dependencies": [],
                "error": str(e)
            }

    def process_json_file(self, json_path: str):
        """
        Charge un fichier JSON de chunks et envoie chaque chunk au LLM.
        """
        log.info(f"[LLM] Début du traitement pour {os.path.basename(json_path)}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"[LLM] Erreur lecture {json_path} : {e}")
            return

        chunks = data.get("chunks", [])
        source_file = data.get("source_file", "unknown")
        llm_results = []

        system_prompt = f"""Tu es un expert en Cyber Threat Intelligence (CTI).
Ta tâche est d'extraire les entités, les relations et les dépendances temporelles de ce texte.
Tu DOIS impérativement utiliser les types d'entités et de relations suivants. N'invente rien.

{self.rocade_context}

--- RÈGLES TEMPORELLES ---
Pour la chronologie, utilise le champ "temporal_dependencies" avec les relations : "BEFORE", "AFTER", ou "SIMULTANEOUS". Extraits l'indicateur de temps exact (ex: "ensuite", "le 12 mars").

Réponds UNIQUEMENT au format JSON. Structure attendue :
{{"entities": [], "relations": [], "temporal_dependencies": []}}
"""

        for i, chunk in enumerate(chunks, 1):
            full_prompt = f"{system_prompt}\n--- TEXTE ---\n{chunk}"

            # Appel au modèle
            response_data = self.call_llm(chunk, full_prompt)

            llm_results.append({
                "chunk_index": i,
                "chunk_preview": chunk[:60].replace("\n", " "),
                "llm_output": response_data
            })

        # Sauvegarde du résultat final de l'extraction
        base_name = os.path.basename(json_path).replace("_processed.json", "_graph_data.json")
        out_path = os.path.join(self.output_dir, base_name)

        payload = {
            "source_file": source_file,
            "n_chunks_processed": len(chunks),
            "graph_data": llm_results
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        log.info(f"[LLM] ✓ Extraction terminée et sauvegardée : {out_path}")

    def load_rocade_schema(self, json_path: str) -> str:
        """
        Lit le JSON de l'ontologie ROCADE (export UML) et génère un texte de contexte épuré.
        Recherche récursivement tous les objets de type 'Class' et 'Relation'.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                rocade_data = json.load(f)

            entities = set()
            relations = set()

            # Fonction interne récursive pour fouiller tout l'arbre JSON
            def extract_elements(obj):
                if isinstance(obj, dict):
                    obj_type = obj.get("type")
                    obj_name = obj.get("name")

                    # Extraction des Entités (Class)
                    if obj_type == "Class" and obj_name and obj_name.strip():
                        entities.add(f"- {obj_name.strip()}")

                    # Extraction des Relations (Relation)
                    elif obj_type == "Relation" and obj_name and obj_name.strip():
                        relations.add(f"- {obj_name.strip()}")

                    # On continue de creuser dans les dictionnaires enfants
                    for key, value in obj.items():
                        extract_elements(value)

                elif isinstance(obj, list):
                    # On continue de creuser dans les listes d'enfants
                    for item in obj:
                        extract_elements(item)

            # Lancement de l'extraction sur la racine du fichier
            extract_elements(rocade_data)

            # Tri alphabétique pour un prompt constant et propre
            sorted_entities = sorted(list(entities))
            sorted_relations = sorted(list(relations))

            # Construction du contexte pour le LLM
            context = "--- ONTOLOGIE ROCADE AUTORISÉE ---\n"
            context += "TYPES D'ENTITÉS :\n" + "\n".join(sorted_entities) + "\n\n"
            context += "TYPES DE RELATIONS :\n" + "\n".join(sorted_relations) + "\n"

            return context

        except Exception as e:
            log.error(f"Erreur lors du chargement de l'ontologie ROCADE : {e}")
            return "Erreur: Ontologie non chargée."