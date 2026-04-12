import os
import re
import json
import glob
import time
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION ET SÉCURITÉ
# ==========================================

# Charge les variables du fichier .env
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("ERREUR CRITIQUE : La variable d'environnement GOOGLE_API_KEY est introuvable.")

genai.configure(api_key=GOOGLE_API_KEY)

# Utilisation du modèle le plus récent et performant pour l'extraction
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# --- CONFIGURATION DU MODE ---
ENABLE_TEST_FILTER = True  # Mettez sur False pour analyser TOUT le rapport (Mode Production)
TEST_CHUNK_LIMIT = 10      # Nombre de chunks "juteux" à analyser en mode test

# --- CHEMINS ---
BASE_PATH = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main"
CHUNKS_DIR = os.path.join(BASE_PATH, "OutputChunks")
OUTPUT_DIR = os.path.join(BASE_PATH, "ExtractedResults")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def get_latest_chunk_file(directory: str) -> str:
    """Scanne le dossier et retourne le fichier JSON de chunks le plus récent."""
    search_pattern = os.path.join(directory, "corpus_chunks_*.json")
    list_of_files = glob.glob(search_pattern)

    if not list_of_files:
        raise FileNotFoundError(f"Aucun fichier de chunks trouvé dans : {directory}. Lancez la Phase 1 d'abord.")

    # Trie par date de modification OS
    return max(list_of_files, key=os.path.getmtime)


def get_juicy_chunks(chunks_list: list, limit: int = 10) -> list:
    """Filtre heuristique : retourne les chunks les plus denses en vocabulaire cyber."""
    cti_keywords = [
        r"\bapt\d+\b", r"unc\d+", "malware", "ransomware", "phishing", r"cve-\d{4}-\d+",
        "vulnerability", "exploit", "payload", "lateral movement", "exfiltration",
        "threat actor", "backdoor", "c2", "command and control", "credential",
        "bypass", "spear-phishing", "cobalt strike", "dropper", "execution"
    ]
    pattern = re.compile("|".join(cti_keywords), re.IGNORECASE)

    scored_chunks = []
    for chunk in chunks_list:
        text = chunk.get("text", "")
        matches = pattern.findall(text)
        score = len(matches)
        if score > 0:
            scored_chunks.append((score, chunk, list(set(matches))))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    print(f"\n[Filtre Heuristique] {len(scored_chunks)} chunks potentiellement intéressants trouvés.")

    top_chunks = []
    for i, item in enumerate(scored_chunks[:limit]):
        score, chunk, keywords = item
        print(f"-> Chunk ID {chunk['chunk_id']} retenu (Score: {score}) - Mots-clés : {keywords}")
        top_chunks.append(chunk)

    return top_chunks


def parse_json_from_response(raw_text: str):
    """Extrait le JSON d'une réponse brute (gère le markdown et le format CoT)."""
    if not raw_text:
        return None

    # 1. Si on est en mode CoT, on cherche uniquement dans les balises <json>
    match = re.search(r'<json>(.*?)</json>', raw_text, re.DOTALL)
    if match:
        clean_text = match.group(1).strip()
    else:
        # 2. Sinon (mode Few-Shot), on nettoie les backticks markdown
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean_text)
    except Exception as e:
        print(f"Erreur de parsing JSON. Texte brut:\n{raw_text[:200]}...")
        return {"error": "JSON parsing failed", "exception": str(e), "raw": raw_text}


def call_gemini(prompt: str):
    """Envoie le prompt à l'API Gemini avec gestion d'erreurs basique."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Erreur API Gemini : {e}")
        return None


# ==========================================
# PROMPTS (ONTOLOGIE ROCADE)
# ==========================================

def get_rocade_few_shot_prompt(text: str, chunk_id: int) -> str:
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst. 
Your task is to extract cybersecurity entities and their semantic and temporal relations from incident reports to populate the ROCADE ontology.
You must output ONLY valid JSON. Do not include any introductory or concluding text.

### 1. ONTOLOGY DEFINITIONS (STRICT ADHERENCE REQUIRED)

#### Allowed Entity Types:
* Threat_Actor: The individual, group, or campaign conducting the attack (e.g., "APT29", "the attackers", "UNC3944").
* Attack_Pattern: A specific tactic or technique performed by the attacker (e.g., "spear-phishing", "lateral movement"). Maps to MITRE ATT&CK concepts.
* Malware: Malicious software, scripts, or payloads (e.g., "Trickbot", "ransomware", "webshell").
* Tool: Legitimate software or administrative utilities repurposed for malicious use (e.g., "PowerShell", "PsExec", "Cobalt Strike").
* Vulnerability: A flaw or weakness exploited in the attack, including specific CVEs (e.g., "CVE-2024-1234").
* Attacker_Infrastructure: External infrastructure controlled by the threat actor (e.g., "C2 server", "malicious domain", "attacker IP").
* Victim_Asset: The internal targets of the attack, such as user accounts, hosts, processes, or internal networks (e.g., "domain controller", "admin account", "lsass.exe").
* Observable: Specific technical artifacts left behind (e.g., "MD5 hash", "specific registry key", "malicious file name").

#### Allowed Relations:
**Semantic (Causal/Structural):**
* USES: A Threat_Actor or Attack_Pattern leverages a Tool, Malware, or Attacker_Infrastructure.
* TARGETS: A Threat_Actor, Malware, or Attack_Pattern aims at a Victim_Asset.
* EXPLOITS: A Threat_Actor, Malware, or Attack_Pattern takes advantage of a Vulnerability.
* INDICATES: An Observable is a technical proof of a Malware, Tool, or Attack_Pattern.

**Temporal:**
* BEFORE: The source entity occurred chronologically prior to the target entity.
* SIMULTANEOUS: The source and target entities occurred at the same time.

### 2. EXTRACTION GUARDRAILS
1. Explicit Mentions Only: Do not infer entities that are not explicitly written in the text.
2. Exact Text Spans: The `mention` field must be an exact substring extracted from the text.
3. ID Naming Convention: You MUST prefix all entity IDs with "C{chunk_id}_" (e.g., C{chunk_id}_E1).
4. Dual Relations: Entities can have both semantic and temporal relations simultaneously.

### 3. EXAMPLES

#### Example 1
Input Text: "The threat actor used PowerShell to execute a lateral movement attack against the domain controller."
Output: 
{{
  "entities": [
    {{"id": "C{chunk_id}_E1", "type": "Threat_Actor", "mention": "threat actor"}},
    {{"id": "C{chunk_id}_E2", "type": "Tool", "mention": "PowerShell"}},
    {{"id": "C{chunk_id}_E3", "type": "Attack_Pattern", "mention": "lateral movement"}},
    {{"id": "C{chunk_id}_E4", "type": "Victim_Asset", "mention": "domain controller"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E2", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "TARGETS"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "BEFORE"}}
  ]
}}

### TEXT TO ANALYZE
{text}
"""

def get_cot_prompt(text: str, chunk_id: int) -> str:
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst specializing in attack kill-chain reconstruction.
Your task is to extract a chronology of events from the provided text using the ROCADE ontology schema.

### 1. ONTOLOGY DEFINITIONS
Allowed Entity Types: Threat_Actor, Attack_Pattern, Malware, Tool, Vulnerability, Attacker_Infrastructure, Victim_Asset, Observable.
Allowed Relations: 
- Semantic: USES, TARGETS, EXPLOITS, INDICATES.
- Temporal: BEFORE, SIMULTANEOUS.

### 2. STRICT INSTRUCTIONS
Before generating the JSON, you must explain your reasoning step-by-step. 
1. Prefix all entity IDs with "C{chunk_id}_".
2. Structure your response exactly as follows:

<thinking>
1. Entity Identification: List all valid entities found in the text.
2. Temporal Markers: Identify the explicit or implicit words indicating time (e.g., "then", "subsequently", "during").
3. Relation Deduction: Deduce the semantic and temporal relationships using ONLY the allowed relations.
</thinking>
<json>
{{
  "entities": [
    // Your extracted entities here
  ],
  "relations": [
    // Your extracted relations here
  ]
}}
</json>

### TEXT TO ANALYZE
{text}
"""


# ==========================================
# PIPELINE PRINCIPALE (PHASE 2)
# ==========================================

def run_extraction_test(test_mode="FEW_SHOT"):
    print(f"\n{'='*50}")
    print(f"Démarrage de l'extraction (Mode: {test_mode})")
    print(f"{'='*50}")

    INPUT_JSON = get_latest_chunk_file(CHUNKS_DIR)
    print(f"[*] Fichier source : {os.path.basename(INPUT_JSON)}")

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    print(f"[*] Total de chunks bruts : {len(all_chunks)}")

    # --- SÉLECTION DES CHUNKS ---
    if ENABLE_TEST_FILTER:
        print(f"\n [MODE TEST ACTIF] Filtrage heuristique activé : sélection des {TEST_CHUNK_LIMIT} meilleurs chunks.")
        chunks_to_process = get_juicy_chunks(all_chunks, limit=TEST_CHUNK_LIMIT)
    else:
        print(f"\n [MODE PRODUCTION ACTIF] Traitement de l'intégralité des {len(all_chunks)} chunks.")
        chunks_to_process = all_chunks

    if not chunks_to_process:
        print("Aucun chunk à traiter. Arrêt.")
        return

    results = []

    # --- BOUCLE D'EXTRACTION ---
    for i, chunk in enumerate(chunks_to_process):
        actual_chunk_id = chunk['chunk_id']
        print(f"\nAnalyse du chunk {i+1}/{len(chunks_to_process)} (ID d'origine: {actual_chunk_id})...")

        # Sélection du prompt
        if test_mode == "FEW_SHOT":
            prompt = get_rocade_few_shot_prompt(chunk['text'], actual_chunk_id)
        else:
            prompt = get_cot_prompt(chunk['text'], actual_chunk_id)

        # Appel LLM et Parsing
        raw_response = call_gemini(prompt)
        parsed_data = parse_json_from_response(raw_response)

        # Sauvegarde en mémoire
        results.append({
            "chunk_metadata": {
                "source": chunk['source'],
                "chunk_id": actual_chunk_id
            },
            "extraction": parsed_data,
            "test_mode": test_mode
        })

        # Pause pour respecter le Rate Limit de l'API gratuite
        time.sleep(4)

        # --- SAUVEGARDE SUR DISQUE ---
    output_file = os.path.join(OUTPUT_DIR, f"results_{test_mode.lower()}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n Extraction terminée avec succès !")
    print(f"Résultats sauvegardés dans : {output_file}")


if __name__ == "__main__":
    # Choisissez le mode à tester en décommentant la ligne appropriée
    run_extraction_test(test_mode="FEW_SHOT")
    # run_extraction_test(test_mode="COT")