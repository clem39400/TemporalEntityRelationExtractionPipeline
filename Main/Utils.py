import re
import json

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

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
        # print(f"-> Chunk ID {chunk['chunk_id']} retenu (Score: {score}) - Mots-clés : {keywords}")
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

#### Example 2
Input Text: "The malicious file 'payload.exe' indicates the presence of Trickbot, which exploited CVE-2024-1234."
Output: 
{{
  "entities": [
    {{"id": "C{chunk_id}_E5", "type": "Observable", "mention": "payload.exe"}},
    {{"id": "C{chunk_id}_E6", "type": "Malware", "mention": "Trickbot"}},
    {{"id": "C{chunk_id}_E7", "type": "Vulnerability", "mention": "CVE-2024-1234"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E6", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "EXPLOITS"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "BEFORE"}}
  ]
}}

### TEXT TO ANALYZE
{text}
"""

def get_cot_prompt(text: str, chunk_id: int) -> str:
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst specializing in highly granular attack kill-chain reconstruction.
Your task is to extract a comprehensive chronology of micro-events from the provided text using the ROCADE ontology schema.

### 1. ONTOLOGY DEFINITIONS
Allowed Entity Types: Threat_Actor, Attack_Pattern, Malware, Tool, Vulnerability, Attacker_Infrastructure, Victim_Asset, Observable.
Allowed Relations: 
- Semantic: USES, TARGETS, EXPLOITS, INDICATES.
- Temporal: BEFORE, SIMULTANEOUS.

### 2. STRICT INSTRUCTIONS
1. DO NOT SUMMARIZE. You must extract every single intermediate step, tool, and asset mentioned.
2. Prefix all entity IDs with "C{chunk_id}_".
3. You must explain your reasoning step-by-step in a <thinking> block before outputting the <json> block.

### 3. EXAMPLE OF EXPECTED GRANULARITY

Input Text: "The threat actor distributed a malicious PDF. When opened, the PDF executed a JavaScript payload which downloaded the Trickbot malware. Trickbot then targeted the local credentials."

<thinking>
1. Entity Identification: I see "threat actor" (Threat_Actor), "malicious PDF" (Observable), "JavaScript payload" (Malware), "Trickbot" (Malware), and "local credentials" (Victim_Asset).
2. Relation Deduction: 
   - The actor USES the PDF.
   - The PDF INDICATES the JavaScript payload.
   - The JavaScript USES Trickbot.
   - Trickbot TARGETS local credentials.
3. Temporal Markers: "When opened" and "then" imply a strict sequence.
   - PDF distribution BEFORE JavaScript execution.
   - JavaScript execution BEFORE Trickbot download.
   - Trickbot download BEFORE targeting credentials.
</thinking>
<json>
{{
  "entities": [
    {{"id": "C{chunk_id}_E1", "type": "Threat_Actor", "mention": "threat actor"}},
    {{"id": "C{chunk_id}_E2", "type": "Observable", "mention": "malicious PDF"}},
    {{"id": "C{chunk_id}_E3", "type": "Malware", "mention": "JavaScript payload"}},
    {{"id": "C{chunk_id}_E4", "type": "Malware", "mention": "Trickbot"}},
    {{"id": "C{chunk_id}_E5", "type": "Victim_Asset", "mention": "local credentials"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E2", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E4", "target": "C{chunk_id}_E5", "relation_type": "TARGETS"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E4", "target": "C{chunk_id}_E5", "relation_type": "BEFORE"}}
  ]
}}
</json>

### TEXT TO ANALYZE
{text}
"""