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
# PROMPTS AVEC ROCADE (STRICT)
# ==========================================

def get_rocade_few_shot_prompt(text: str, chunk_id: int) -> str:
    """Prompt Few-Shot ROCADE rééquilibré : sémantique réaliste et chaîne temporelle fidèle."""
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst.
Your task is to extract an accurate, highly granular Cyber Knowledge Graph from the text strictly adhering to the ROCADE ontology schema.

### 1. ROCADE ONTOLOGY SCHEMA
- **Entity Types:** Threat_Actor, Attack_Pattern, Malware, Tool, Attacker_Infrastructure, Victim_Asset, Observable, Vulnerability.
- **Allowed Relations:** 
  - Semantic: USES, TARGETS, EXPLOITS, INDICATES.
  - Temporal: BEFORE, SIMULTANEOUS.

### 2. EXTRACTION GUIDELINES
1. **Semantic Roles:**
   - A `Threat_Actor` `USES` Attack_Patterns, Malware, or Tools explicitly operated by the group.
   - Delivery mechanisms and artifacts `INDICATE` downloaded payloads or observables.
   - Malware and Tools `TARGET` Victim_Assets (e.g., credentials, servers, databases).
2. **Temporal Kill-Chain (`BEFORE`):**
   - Connect distinct, consecutive steps of the attack lifecycle using `BEFORE` relations (e.g., Initial Access -> Payload Download -> Execution -> Lateral Movement -> Exfiltration).
   - Do NOT chain unrelated parallel tools with BEFORE unless a sequential execution is explicitly stated.
3. **Exact Substring Mentions:** The "mention" field must be an exact substring from the source text.
4. **Namespace ID:** All entity IDs MUST start with "C{chunk_id}_".
5. **Output Format:** Output ONLY a single valid JSON object. No conversational intro, no markdown text outside the JSON.

### 3. GOLD STANDARD EXAMPLES

Example 1 (Initial Access & Compromise):
Input Text: "The threat actor sent spear phishing emails containing a hyperlink. The link downloaded a malicious executable, which dropped the WEBC2-TABLE backdoor."
Output:
{{
  "entities": [
    {{"id": "C{chunk_id}_E1", "type": "Threat_Actor", "mention": "threat actor"}},
    {{"id": "C{chunk_id}_E2", "type": "Attack_Pattern", "mention": "spear phishing emails"}},
    {{"id": "C{chunk_id}_E3", "type": "Observable", "mention": "hyperlink"}},
    {{"id": "C{chunk_id}_E4", "type": "Observable", "mention": "malicious executable"}},
    {{"id": "C{chunk_id}_E5", "type": "Malware", "mention": "WEBC2-TABLE"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E2", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E4", "target": "C{chunk_id}_E5", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E4", "target": "C{chunk_id}_E5", "relation_type": "BEFORE"}}
  ]
}}

Example 2 (Lateral Movement & Exfiltration):
Input Text: "The attackers used legitimate credentials to access the network. They then executed psexec to compromise the domain controller and used RAR to compress files for exfiltration."
Output:
{{
  "entities": [
    {{"id": "C{chunk_id}_E6", "type": "Threat_Actor", "mention": "attackers"}},
    {{"id": "C{chunk_id}_E7", "type": "Victim_Asset", "mention": "legitimate credentials"}},
    {{"id": "C{chunk_id}_E8", "type": "Victim_Asset", "mention": "network"}},
    {{"id": "C{chunk_id}_E9", "type": "Tool", "mention": "psexec"}},
    {{"id": "C{chunk_id}_E10", "type": "Victim_Asset", "mention": "domain controller"}},
    {{"id": "C{chunk_id}_E11", "type": "Tool", "mention": "RAR"}},
    {{"id": "C{chunk_id}_E12", "type": "Victim_Asset", "mention": "files"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E7", "target": "C{chunk_id}_E8", "relation_type": "TARGETS"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E9", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E9", "target": "C{chunk_id}_E10", "relation_type": "TARGETS"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E11", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E11", "target": "C{chunk_id}_E12", "relation_type": "TARGETS"}},
    {{"source": "C{chunk_id}_E7", "target": "C{chunk_id}_E9", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E9", "target": "C{chunk_id}_E11", "relation_type": "BEFORE"}}
  ]
}}

### TEXT TO ANALYZE
{text}
"""

def get_rocade_cot_prompt(text: str, chunk_id: int) -> str:
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst specializing in highly granular attack kill-chain reconstruction.
Your task is to extract a comprehensive chronology of micro-events from the provided text using the ROCADE ontology schema.

### 1. ONTOLOGY DEFINITIONS
Allowed Entity Types: Threat_Actor, Attack_Pattern, Malware, Tool, Vulnerability, Attacker_Infrastructure, Victim_Asset, Observable.
Allowed Relations: 
- Semantic: USES, TARGETS, EXPLOITS, INDICATES.
- Temporal: BEFORE, SIMULTANEOUS.

### 2. STRICT EXTRACTION RULES
1. Comprehensive but Precise: Extract all technical entities (tools, malware, assets, actors). Do not invent generic entities.
2. Strict Chronological Chaining: Reconstruct the kill-chain by linking strictly consecutive actions with BEFORE relations (e.g., Step A -> Step B -> Step C). Do not create branching or duplicate timelines unless explicitly stated.
3. No Hallucinated Links: Only create semantic relations (USES, TARGETS) if the text explicitly describes the interaction. Do not connect every single entity to the Threat_Actor.
4. ID Prefix: All entity IDs MUST start with "C{chunk_id}_".

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


# ==========================================
# PROMPTS SANS ROCADE (LIBRES)
# ==========================================

def get_no_rocade_few_shot_prompt(text: str, chunk_id: int) -> str:
    """Prompt Few-Shot SANS contraintes ontologiques (Coréférence, Anti-Pattern & 2 Exemples)."""
    return f"""You are a cybersecurity analyst.
Your task is to extract entities and their relationships from the text to build a graph.
You are completely FREE to invent ANY Entity Type (e.g., "Hacker", "IP_Address", "Software") and ANY Relation Type (e.g., "HACKS", "DOWNLOADS", "OCCURS_AFTER") that you think best describes the text.

### STRICT RULES - THE "LINEAR KILL-CHAIN" METHOD
1. You MUST output ONLY a valid JSON object. No markdown formatting outside the JSON, and no explanations.
2. Linear Topology (Crucial): DO NOT build a "star" graph where the Attacker connects to every single tool directly. Instead, build a LINEAR chain following the exact sequential flow of the attack (e.g., Attacker -> Tool A -> Tool B -> Target).
3. Continuous Time Chain: connect sequential steps with an unbroken chain of a temporal relation of your choice (e.g., HAPPENS_BEFORE).
4. Entity Resolution (Coreference): if the same real-world entity reappears later via a pronoun or synonym (e.g., "they" after "the attacker", "it" after a named tool), reuse the SAME entity id — do NOT create a duplicate entity for the same referent.
5. All entity IDs MUST start with "C{chunk_id}_" (e.g., C{chunk_id}_E1).
6. The "mention" field must be an exact substring from the text (the noun phrase, not the pronoun, unless no earlier noun phrase exists).

### EXAMPLES

Example 1 (Coreference):
Input Text: "The attacker breached the web server using SQLmap. Afterwards, they exfiltrated the database using it."
Output:
{{
  "entities": [
    {{"id": "C{chunk_id}_E1", "type": "Attacker", "mention": "attacker"}},
    {{"id": "C{chunk_id}_E2", "type": "Hacking_Tool", "mention": "SQLmap"}},
    {{"id": "C{chunk_id}_E3", "type": "Server", "mention": "web server"}},
    {{"id": "C{chunk_id}_E4", "type": "Data", "mention": "database"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E2", "relation_type": "UTILIZES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "BREACHES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "PRECEDES_EXFILTRATION_OF"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "HAPPENS_BEFORE"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "HAPPENS_BEFORE"}}
  ]
}}
Note: "they" refers back to E1 ("attacker") and "it" refers back to E2 ("SQLmap") — neither was re-created as a new entity.

Example 2 (Longer chain):
Input Text: "Afterwards, the malware executed a powershell script, which downloaded a second-stage payload and exfiltrated the database."
Output:
{{
  "entities": [
    {{"id": "C{chunk_id}_E5", "type": "Malicious_Code", "mention": "malware"}},
    {{"id": "C{chunk_id}_E6", "type": "Script", "mention": "powershell script"}},
    {{"id": "C{chunk_id}_E7", "type": "Malicious_Code", "mention": "second-stage payload"}},
    {{"id": "C{chunk_id}_E8", "type": "Data", "mention": "database"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E6", "relation_type": "EXECUTES"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "DOWNLOADS"}},
    {{"source": "C{chunk_id}_E7", "target": "C{chunk_id}_E8", "relation_type": "STEALS"}},
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E6", "relation_type": "OCCURS_BEFORE"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "OCCURS_BEFORE"}},
    {{"source": "C{chunk_id}_E7", "target": "C{chunk_id}_E8", "relation_type": "OCCURS_BEFORE"}}
  ]
}}

### ANTI-PATTERN — WHAT NOT TO DO (using Example 2's text)
Incorrect Output (rejected — "star" topology, do NOT imitate this):
{{
  "relations": [
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E6", "relation_type": "EXECUTES"}},
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E7", "relation_type": "DOWNLOADS"}},
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E8", "relation_type": "STEALS"}}
  ]
}}
Why this is wrong: every edge starts from the malware (E5) directly, ignoring the actual chain malware -> script -> payload -> database. This destroys the sequential order and must be avoided.

### FINAL CHECKLIST (verify before you output)
- [ ] No entity has more than one direct downstream edge (no star pattern).
- [ ] Every consecutive step is linked by a temporal relation.
- [ ] No two entities represent the same real-world referent.
- [ ] Output is a single valid JSON object — no markdown fences, no trailing commas.

### TEXT TO ANALYZE
{text}
"""

def get_no_rocade_cot_prompt(text: str, chunk_id: int) -> str:
    """Prompt Chain-of-Thought SANS ROCADE : structure identique à la version ROCADE avec liberté de taxonomie."""
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst specializing in highly granular attack kill-chain reconstruction.
Your task is to extract a comprehensive chronology of micro-events from the provided text to build a knowledge graph.

### 1. TAXONOMY & ONTOLOGY FREEDOM
* You have complete freedom: you may use standard cybersecurity categories (e.g., Threat_Actor, Attack_Pattern, Malware, Tool, Victim_Asset, USES, TARGETS, BEFORE) OR invent ANY custom Entity Types and Relation Types that you see fit.

### 2. STRICT EXTRACTION RULES
1. Comprehensive but Precise: Extract all technical entities (tools, malware, assets, actors). Do not invent generic entities.
2. Strict Chronological Chaining: Reconstruct the kill-chain by linking strictly consecutive actions with temporal relations such as BEFORE (e.g., Step A -> Step B -> Step C). Do not create branching or duplicate timelines unless explicitly stated.
3. No Hallucinated Links: Only create semantic relations if the text explicitly describes the interaction. Do not connect every single entity to the Threat_Actor.
4. ID Prefix: All entity IDs MUST start with "C{chunk_id}_".

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