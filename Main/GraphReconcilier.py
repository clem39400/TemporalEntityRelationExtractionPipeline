import json
import os
import re
from sentence_transformers import SentenceTransformer, util

REGEX_STRICT_ENTITIES = [
    re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE),
    re.compile(r"^(UNC|APT|FIN)\d+$", re.IGNORECASE)
]

def requires_strict_match(mention: str) -> bool:
    mention_clean = mention.strip()
    return any(regex.search(mention_clean) for regex in REGEX_STRICT_ENTITIES)

def reconcile_graph(input_path, output_path, similarity_threshold=0.85, malware_threshold=0.95):
    if not os.path.exists(input_path):
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)

    # MOTEUR CYBER SÉMANTIQUE
    print("[*] Chargement du modèle de similarité sémantique (basel/ATTACK-BERT)...")
    model = SentenceTransformer('basel/ATTACK-BERT')

    global_entities = {}
    global_relations = []
    local_to_global_map = {}
    global_id_counter = 1

    for entry in extracted_data:
        if not entry.get("extraction"): continue

        chunk_id = entry["chunk_metadata"].get("chunk_id", "unknown")
        entities = entry["extraction"].get("entities", [])
        relations = entry["extraction"].get("relations", [])

        # 1. Résolution d'Entités (Entity Resolution)
        for ent in entities:
            local_id = ent.get("id", "")
            ent_type = ent.get("type", "Unknown")
            mention = ent.get("mention", ent.get("labels", [""])[0] if isinstance(ent.get("labels"), list) else "")

            if not mention or not local_id: continue

            mention = str(mention).strip()
            mention_lower = mention.lower()
            is_strict_local = requires_strict_match(mention)
            ent_emb = model.encode(mention_lower, convert_to_tensor=True)

            matched_global_id = None
            for g_id, g_data in global_entities.items():
                if g_data["type"] == ent_type:
                    is_match = False
                    if is_strict_local or g_data["is_strict"]:
                        if mention_lower in g_data["mentions_lower"]: is_match = True
                    else:
                        sim = util.cos_sim(ent_emb, g_data["embedding"]).item()
                        current_threshold = malware_threshold if ent_type == "Malware" else similarity_threshold
                        if mention_lower in g_data["mentions_lower"] or sim >= current_threshold:
                            is_match = True

                    if is_match:
                        matched_global_id = g_id
                        g_data["mentions"].add(mention)
                        g_data["mentions_lower"].add(mention_lower)
                        break

            if matched_global_id:
                local_to_global_map[local_id] = matched_global_id
            else:
                new_g_id = f"G_E{global_id_counter}"
                global_entities[new_g_id] = {
                    "type": ent_type, "mentions": {mention}, "mentions_lower": {mention_lower},
                    "embedding": ent_emb, "is_strict": is_strict_local
                }
                local_to_global_map[local_id] = new_g_id
                global_id_counter += 1

        # 2. Migration des Relations
        seen_edges = set()
        for rel in relations:
            src_local = rel.get("source", "")
            tgt_local = rel.get("target", "")
            src_global = local_to_global_map.get(src_local)
            tgt_global = local_to_global_map.get(tgt_local)
            rel_type = rel.get("relation_type", "UNKNOWN").upper()

            if src_global and tgt_global:
                edge_sig = (src_global, rel_type, tgt_global)
                if edge_sig not in seen_edges:
                    seen_edges.add(edge_sig)
                    global_relations.append({
                        "source": src_global, "target": tgt_global,
                        "relation_type": rel_type, "provenance": f"chunk_{chunk_id}"
                    })

    final_nodes = [{"id": k, "type": v["type"], "labels": list(v["mentions"])} for k, v in global_entities.items()]
    final_graph = {"metadata": {"total_nodes": len(final_nodes), "total_relations": len(global_relations)}, "nodes": final_nodes, "edges": global_relations}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_graph, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    pass