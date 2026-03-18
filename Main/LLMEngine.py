"""
LLMEngine.py
------------
Module d'inférence LLM optimisé pour Ollama local (Qwen 2.5 14B).
Extraction d'entités et relations avec contraintes de schéma strictes et cache (API Chat).
"""

import json
import os
import logging
import requests
import re

log = logging.getLogger(__name__)

class LLMEngine:
    # On définit Qwen 2.5 14B comme modèle par défaut
    def __init__(self, output_dir: str, rocade_json_path: str, model_name: str = "qwen2.5:14b"):
        self.output_dir = output_dir
        self.model_name = model_name
        # Utilisation de l'API Chat locale pour profiter du cache du prompt système
        self.api_url = "http://localhost:11434/api/chat"
        os.makedirs(self.output_dir, exist_ok=True)

        log.info(f"[LLM] Chargement de l'ontologie depuis {rocade_json_path}...")
        self.rocade_context = self.load_rocade_schema(rocade_json_path)

    def call_llm(self, chunk_text: str, system_prompt: str) -> dict:
        """Appel à l'API locale d'Ollama avec forçage JSON et température à 0."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Voici le texte à analyser :\n{chunk_text}"}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0 # Empêche l'hallucination de relations
            }
        }

        try:
            # Timeout fixé à 180s pour laisser le temps au modèle 14B de répondre
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("message", {}).get("content", "")

            try:
                return json.loads(generated_text)
            except json.JSONDecodeError:
                # Tentative de récupération si le JSON est mal formatté
                match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                return {"entities": [], "relations": [], "temporal_dependencies": [], "error": "JSON_DECODE_ERROR"}

        except Exception as e:
            log.error(f"[LLM] Erreur lors de l'appel à Ollama ({self.model_name}) : {e}")
            return {"entities": [], "relations": [], "temporal_dependencies": [], "error": str(e)}

    def process_json_file(self, json_path: str):
        """Traite un fichier de chunks via Ollama (Qwen 2.5)."""
        log.info(f"[LLM] Début de l'extraction pour {os.path.basename(json_path)}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"[LLM] Erreur lecture {json_path} : {e}")
            return

        chunks = data.get("chunks", [])
        source_file = data.get("source_file", "unknown")
        llm_results = []

        system_prompt = f"""Tu es un expert en Cyber Threat Intelligence spécialisé dans l'ontologie ROCADE.
Ta mission est d'extraire les entités et les relations à partir de rapports techniques.

{self.rocade_context}

--- GESTION DES TABLEAUX ---
Si le texte contient un tableau Markdown (|---|), analyse chaque ligne comme une relation potentielle. 

--- EXEMPLE DE RÉFÉRENCE (FEW-SHOT) ---
Texte : "L'attaquant APT28 a utilisé le malware X-Agent pour cibler le Parlement allemand en juin 2023."
Réponse attendue :
{{
  "entities": [
    {{"id": "e1", "mention": "APT28", "rocade_type": "Attacker"}},
    {{"id": "e2", "mention": "X-Agent", "rocade_type": "Threat Capability"}},
    {{"id": "e3", "mention": "Parlement allemand", "rocade_type": "Object at Risk"}}
  ],
  "relations": [
    {{"subject_id": "e1", "predicate": "uses", "object_id": "e2"}},
    {{"subject_id": "e1", "predicate": "targets", "object_id": "e3"}}
  ],
  "temporal_dependencies": [
    {{"event_id": "e1", "temporal_relation": "SIMULTANEOUS", "related_event_id": "e3", "time_indicator": "en juin 2023"}}
  ]
}}

--- CONSIGNES STRICTES (RÈGLES ABSOLUES) ---
1. ONTOLOGIE OBLIGATOIRE : Tu DOIS utiliser EXCLUSIVEMENT les types d'entités et les relations exactes listés dans l'ONTOLOGIE ROCADE ci-dessus. N'invente AUCUN verbe.
2. INTÉGRITÉ DES IDs : Les valeurs 'subject_id' et 'object_id' dans les relations DOIVENT correspondre à un 'id' (ex: 'e1') déclaré dans la liste 'entities'.
3. Le champ 'mention' doit être une copie exacte du texte.
4. Pour les dépendances temporelles, utilise : BEFORE, AFTER, ou SIMULTANEOUS.
5. Réponds uniquement par un objet JSON. Si aucune menace n'est détectée, renvoie des listes vides.
"""

        for i, chunk in enumerate(chunks, 1):
            log.info(f"[LLM] Inférence chunk {i}/{len(chunks)} via Qwen 2.5 14B...")
            response_data = self.call_llm(chunk_text=chunk, system_prompt=system_prompt)

            llm_results.append({
                "chunk_index": i,
                "chunk_preview": chunk[:60].replace("\n", " "),
                "llm_output": response_data
            })

        base_name = os.path.basename(json_path).replace("_processed.json", "_graph_data.json")
        out_path = os.path.join(self.output_dir, base_name)

        payload = {"source_file": source_file, "graph_data": llm_results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        log.info(f"[LLM] ✓ Extraction terminée : {out_path}")

    def load_rocade_schema(self, json_path: str) -> str:
        """Extrait récursivement les classes et relations du fichier ROCADE JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                rocade_data = json.load(f)

            entities, relations = set(), set()

            def extract_elements(obj):
                if isinstance(obj, dict):
                    name = obj.get("name")
                    if obj.get("type") == "Class" and name:
                        entities.add(f"- {name.strip()}")
                    elif obj.get("type") == "Relation" and name:
                        relations.add(f"- {name.strip()}")
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            extract_elements(value)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            extract_elements(item)

            extract_elements(rocade_data)

            return ("--- ONTOLOGIE ROCADE AUTORISÉE ---\n"
                    "TYPES D'ENTITÉS :\n" + "\n".join(sorted(entities)) + "\n\n"
                                                                          "TYPES DE RELATIONS :\n" + "\n".join(sorted(relations)) + "\n")

        except Exception as e:
            log.error(f"Erreur chargement ROCADE : {e}")
            return "Erreur: Ontologie non chargée."