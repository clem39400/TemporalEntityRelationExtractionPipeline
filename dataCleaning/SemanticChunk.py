import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ── Modèles globaux ───────────────────────────────────────────────────────────
# Chargés une seule fois au niveau module pour éviter de les recharger à chaque appel.
# Pour un meilleur domaine CTI, remplacer par :
#   - nlp   : "en_core_web_trf" (transformer) ou un NER custom CTI
#   - model : "BAAI/bge-m3" (multilingue, 8192 tokens, utilisé dans SC-LKM)
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


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


# ── Algorithme principal ──────────────────────────────────────────────────────

def semantic_chunking_improved(
        paragraphs: list[str],
        theta_s: float = 0.5,
        theta_e: float = 0.1,
        l_max: int = 400,
) -> list[str]:
    """
    Chunking sémantique en deux dimensions (SC-LKM, Wang et al., 2025).

    Paramètres
    ----------
    paragraphs : list[str]
        Paragraphes issus du Stage 1 (découpage structurel).
    theta_s : float
        Seuil de similarité cosinus sous lequel une coupure est déclenchée.
    theta_e : float
        Seuil d'overlap d'entités (Jaccard) sous lequel une coupure est déclenchée.
    l_max : int
        Longueur maximale d'un chunk en mots (approximation tokens LLM).

    Retourne
    --------
    list[str] : chunks sémantiquement cohérents séparés par double saut de ligne.

    """
    if not paragraphs:
        return []

    chunks = []
    current_chunk: list[str] = []
    current_entities: list[str] = []  # entités du dernier paragraphe (fenêtre = 1)
    current_length: int = 0
    current_centroid = None
    last_emb = None
    n_in_chunk: int = 0

    # Pré-calcul de tous les embeddings en un seul appel (efficacité GPU/CPU)
    embeddings = model.encode(paragraphs, convert_to_tensor=True)

    for i, p_i in enumerate(paragraphs):
        doc = nlp(p_i)
        l_i = len(p_i.split())  # longueur en mots
        entities_i = get_entities(doc)
        emb_i = embeddings[i]

        # ── Calcul des scores ─────────────────────────────────────────────────
        if current_centroid is None:
            # Premier paragraphe du chunk : toujours accepté
            s_i, e_i = 1.0, 1.0
        else:
            s_centroid = util.cos_sim(emb_i, current_centroid).item()
            s_last = util.cos_sim(emb_i, last_emb).item()
            s_i = 0.6 * s_centroid + 0.4 * s_last  # similarité mixte
            e_i = jaccard_weighted(entities_i, current_entities)  # vs dernier §

        # ── Condition de coupure ──────────────────────────────────────────────
        if current_chunk and (
                s_i < theta_s
                or e_i < theta_e
                or current_length + l_i > l_max
        ):
            chunks.append("\n\n".join(current_chunk))
            # Reset complet
            current_chunk = []
            current_entities = []
            current_length = 0
            current_centroid = None
            last_emb = None
            n_in_chunk = 0

        # ── Ajout du paragraphe au chunk courant ──────────────────────────────
        current_chunk.append(p_i)
        current_entities = entities_i  # fenêtre glissante : dernier § seulement
        current_length += l_i
        last_emb = emb_i
        n_in_chunk += 1

        # ── Mise à jour du centroïde (moyenne exacte) ─────────────────────────
        if current_centroid is None:
            current_centroid = emb_i.clone()
        else:
            current_centroid = (
                                       current_centroid * (n_in_chunk - 1) + emb_i
                               ) / n_in_chunk

    # Dernier chunk résiduel
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
