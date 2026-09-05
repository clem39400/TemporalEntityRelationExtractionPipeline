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
    """Prompt Few-Shot ROCADE ultime avec Anti-Patterns et Checklist anti-FP."""
    return f"""You are an expert Cyber Threat Intelligence (CTI) analyst specializing in APT campaign mapping and structured graph generation. Extract an exhaustive, highly precise set of entities and relations from the text strictly adhering to the ROCADE ontology schema.

### 1. ROCADE ONTOLOGY SCHEMA & DEFINITIONS
- **Threat_Actor**: Named threat groups or actors (e.g., "APT1").
- **Attack_Pattern**: Tactics, techniques, procedures (e.g., "spear phishing").
- **Malware**: Malicious software or payloads (e.g., "WEBC2-TABLE", "backdoor").
- **Tool**: Legitimate or administrative utilities used offensively (e.g., "psexec", "RAR", "FTP", "at.exe").
- **Attacker_Infrastructure**: External resources, C2 servers, hop points (e.g., "hop points").
- **Victim_Asset**: Compromised internal systems, servers, data (e.g., "domain controller", "Microsoft Exchange Server").
- **Observable**: File names, hashes, links, artifacts (e.g., "hyperlink", "malicious executable").
- **Vulnerability**: Exploited flaws.

**Allowed Relations:**
- Semantic: USES, TARGETS, EXPLOITS, INDICATES.
- Temporal: BEFORE, SIMULTANEOUS.

### 2. STRICT EXTRACTION & TOPOLOGY RULES
1. **Linear Topology (Anti-Star Graph Constraint):** ABSOLUTELY FORBIDDEN to build a "star" graph where the Threat_Actor connects directly to every tool or asset. Build a sequential, step-by-step chain reflecting the attack flow.
2. **Continuous Time Chain:** Connect sequential execution steps with an unbroken chain of `BEFORE` relations mirroring the semantic path.
3. **Coreference / No Duplicate Entities:** If the same real-world entity appears via a synonym or pronoun, reuse the exact same entity ID. Do not re-create duplicates.
4. **Exact String Matching:** The "mention" field must be an exact substring from the text.
5. **Localized Namespace Prefix:** All entity IDs MUST start with "C{chunk_id}_".
6. **Strict Output Format:** Output ONLY a valid JSON object. No conversational intro, no markdown fences outside JSON.

### 3. GOLD STANDARD EXAMPLE
Input Text: "The threat actor sent spear phishing emails containing a hyperlink. The link downloaded a malicious executable, which established a backdoor. The attackers then used psexec to compromise the domain controller."
Output:
{{
  "entities": [
    {{"id": "C{chunk_id}_E1", "type": "Threat_Actor", "mention": "threat actor"}},
    {{"id": "C{chunk_id}_E2", "type": "Attack_Pattern", "mention": "spear phishing emails"}},
    {{"id": "C{chunk_id}_E3", "type": "Observable", "mention": "hyperlink"}},
    {{"id": "C{chunk_id}_E4", "type": "Observable", "mention": "malicious executable"}},
    {{"id": "C{chunk_id}_E5", "type": "Malware", "mention": "backdoor"}},
    {{"id": "C{chunk_id}_E6", "type": "Tool", "mention": "psexec"}},
    {{"id": "C{chunk_id}_E7", "type": "Victim_Asset", "mention": "domain controller"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E2", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E4", "target": "C{chunk_id}_E5", "relation_type": "INDICATES"}},
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E6", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "TARGETS"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E3", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E4", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E4", "target": "C{chunk_id}_E5", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E5", "target": "C{chunk_id}_E6", "relation_type": "BEFORE"}},
    {{"source": "C{chunk_id}_E6", "target": "C{chunk_id}_E7", "relation_type": "BEFORE"}}
  ]
}}

### 4. ANTI-PATTERN — WHAT NOT TO DO
Incorrect Output (Rejected — "star" topology, do NOT connect the Threat Actor directly to everything):
{{
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E4", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E5", "relation_type": "USES"}},
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E6", "relation_type": "USES"}}
  ]
}}
Why this is wrong: The threat actor is linked directly to every single item, creating a star graph instead of a sequential chain. Avoid this.

### 5. FINAL CHECKLIST
- [ ] No star topology (the threat actor does not connect to every downstream tool).
- [ ] Every sequential step has a matching `BEFORE` relation.
- [ ] All IDs start with "C{chunk_id}_".
- [ ] Output is valid JSON only.

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
    """Prompt Chain-of-Thought SANS contraintes ontologiques (Topologie Linéaire Forcée)."""
    return f"""You are a cybersecurity analyst.
Your task is to extract entities and their relationships from the text to build a graph.
You are completely FREE to invent ANY Entity Type and ANY Relation Type that you think best describes the text.

### STRICT RULES - THE "LINEAR KILL-CHAIN" METHOD
1. Linear Topology (Crucial): DO NOT build a "star" graph where the Attacker connects to every single tool. Instead, build a LINEAR chain representing the exact sequential flow of the attack (e.g., Attacker -> Tool A -> Tool B -> Target).
2. Continuous Time Chain: You MUST connect these sequential steps with an unbroken chain of temporal relations (e.g., HAPPENS_BEFORE).
3. Prefix all entity IDs with "C{chunk_id}_". The "mention" field must be an exact substring from the text.
4. You must explain your reasoning step-by-step in a <thinking> block before outputting the <json> block. Explicitly enforce the linear topology in your reasoning.

### EXAMPLE
Input Text: "The attacker breached the web server using SQLmap. Afterwards, they exfiltrated the database."

<thinking>
1. Entities: "attacker" (Attacker), "SQLmap" (Hacking_Tool), "web server" (Server), "database" (Data).
2. Linear Semantic Relations: 
   - Attacker UTILIZES SQLmap. 
   - SQLmap BREACHES web server. 
   - web server COMPROMISES database.
3. Continuous Temporal Chain: 
   - SQLmap HAPPENS_BEFORE web server.
   - web server HAPPENS_BEFORE database.
</thinking>
<json>
{{
  "entities": [
    {{"id": "C{chunk_id}_E1", "type": "Attacker", "mention": "attacker"}},
    {{"id": "C{chunk_id}_E2", "type": "Server", "mention": "web server"}},
    {{"id": "C{chunk_id}_E3", "type": "Hacking_Tool", "mention": "SQLmap"}},
    {{"id": "C{chunk_id}_E4", "type": "Data", "mention": "database"}}
  ],
  "relations": [
    {{"source": "C{chunk_id}_E1", "target": "C{chunk_id}_E3", "relation_type": "UTILIZES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E2", "relation_type": "BREACHES"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E4", "relation_type": "COMPROMISES"}},
    {{"source": "C{chunk_id}_E3", "target": "C{chunk_id}_E2", "relation_type": "HAPPENS_BEFORE"}},
    {{"source": "C{chunk_id}_E2", "target": "C{chunk_id}_E4", "relation_type": "HAPPENS_BEFORE"}}
  ]
}}
</json>

### TEXT TO ANALYZE
{text}
"""