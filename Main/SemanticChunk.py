"""
SemanticChunk.py
----------------
Implémentation du chunking sémantique (SC-LKM) optimisé pour la CTI.
Intègre BAAI/bge-m3 pour l'embedding et un pipeline spaCy enrichi
(EntityRuler) pour la détection spécifique d'entités cyber (CVE, 
Threat Actors, Hashes, IPs, Malwares).
"""

import re
import spacy
from sentence_transformers import SentenceTransformer, util
from MitreWhitelistLoader import MitreWhitelistLoader

# ── Initialisation globale des modèles ──────────────────────────────────────

print("[SemanticChunk] Chargement des modèles NLP et Embedding...")

# 1. Modèle d'embedding
# ASTUCE VITESSE : Si "BAAI/bge-m3" reste trop lent sur ta machine (car très lourd), 
# commente la ligne ci-dessous et décommente la suivante pour utiliser un modèle ultra-léger et rapide.
model = SentenceTransformer("BAAI/bge-m3")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Modèle spaCy allégé (Désactivation de la grammaire pour exploser la vitesse)
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

# 3. Enrichissement CTI via EntityRuler
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler")
    
    cti_patterns = [
        # Vulnérabilités (ex: CVE-2021-40444)
        {"label": "CVE", "pattern": [{"TEXT": {"REGEX": r"(?i)CVE-\d{4}-\d{4,7}"}}]},
        # Hashes (MD5, SHA1, SHA256)
        {"label": "HASH", "pattern": [{"TEXT": {"REGEX": r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"}}]},
        # IP Addresses
        {"label": "IP_ADDR", "pattern": [{"TEXT": {"REGEX": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"}}]},
        # Groupes et malwares (exemples)
        {"label": "THREAT_ACTOR", "pattern": [{"LOWER": "apt28"}]},
        {"label": "THREAT_ACTOR", "pattern": [{"LOWER": "lazarus"}]},
    ]
    
    # Intégration dynamique des Threat Actors depuis le MITRE
    try:
        mitre_loader = MitreWhitelistLoader(ttl_days=7)
        threat_actors = mitre_loader.get_whitelist()
        for actor in threat_actors:
            # Gestion des noms multi-mots (ex: "fancy bear" -> [{"LOWER": "fancy"}, {"LOWER": "bear"}])
            pattern = [{"LOWER": word} for word in actor.lower().split()]
            cti_patterns.append({"label": "THREAT_ACTOR", "pattern": pattern})
    except Exception as e:
        print(f"[SemanticChunk] Avertissement: Impossible de charger la whitelist MITRE ({e})")

    ruler.add_patterns(cti_patterns)


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def get_entities(doc) -> list[str]:
    """Extrait les entités nommées d'un doc spaCy sous forme de liste de strings."""
    return [ent.text.lower() for ent in doc.ents]


def jaccard_weighted(e1: list[str], e2: list[str]) -> float:
    """
    Calcule le Jaccard entre deux listes d'entités.
    Retourne 1.0 si les deux listes sont vides (continuité implicite).
    """
    if not e1 and not e2:
        return 1.0
    set1, set2 = set(e1), set(e2)
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union) if union else 0.0


def _split_at_sentence_boundary(text: str, l_max: int) -> list[str]:
    """
    Redécoupe un chunk trop long (> l_max mots) en sous-chunks de
    longueur <= l_max mots, en s'arrêtant toujours à la fin d'une phrase.
    """
    if len(text.split()) <= l_max:
        return [text]

    sentences = re.split(r'(?<=[.!?]) +', text)
    sub_chunks: list[str] = []
    current: list[str] = []
    current_len: int = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > l_max:
            sub_chunks.append(' '.join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        sub_chunks.append(' '.join(current))

    return sub_chunks


# ── Algorithme principal ──────────────────────────────────────────────────────

def semantic_chunking_improved(
        paragraphs: list[str],
        theta_s: float = 0.5,
        theta_e: float = 0.15,
        l_max: int = 400,
) -> list[str]:
    """
    Chunking sémantique en deux dimensions (SC-LKM, Wang et al., 2025).
    """
    if not paragraphs:
        return []

    chunks = []
    current_chunk: list[str] = []
    current_entities: list[str] = [] 
    current_length: int = 0
    current_centroid = None
    last_emb = None
    n_in_chunk: int = 0

    # OPTIMISATION 1 : Vectorisation par lots (batching) au lieu d'un par un
    embeddings = model.encode(paragraphs, batch_size=32, show_progress_bar=False, convert_to_tensor=True)

    # OPTIMISATION 2 : Traitement NLP spaCy par lots via nlp.pipe() (Gain de temps massif)
    docs = list(nlp.pipe(paragraphs, batch_size=32))

    for i, (p_i, doc) in enumerate(zip(paragraphs, docs)):
        l_i = len(p_i.split())
        entities_i = get_entities(doc)
        emb_i = embeddings[i]

        if current_centroid is None:
            s_i, e_i = 1.0, 1.0
        else:
            s_centroid = util.cos_sim(emb_i, current_centroid).item()
            s_last = util.cos_sim(emb_i, last_emb).item()
            s_i = 0.6 * s_centroid + 0.4 * s_last
            e_i = jaccard_weighted(entities_i, current_entities)

        if current_chunk and (
                s_i < theta_s
                or e_i < theta_e
                or current_length + l_i > l_max
        ):
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_entities = []
            current_length = 0
            current_centroid = None
            last_emb = None
            n_in_chunk = 0

        current_chunk.append(p_i)
        current_entities = entities_i
        current_length += l_i
        last_emb = emb_i
        n_in_chunk += 1

        if current_centroid is None:
            current_centroid = emb_i.clone()
        else:
            current_centroid = (current_centroid * (n_in_chunk - 1) + emb_i) / n_in_chunk

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    final_chunks: list[str] = []
    for c in chunks:
        final_chunks.extend(_split_at_sentence_boundary(c, l_max))

    return final_chunks