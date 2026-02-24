import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np



def semantic_chunking(paragraphs, theta_s=0.5, theta_e=0.1, l_max=400):
    chunks = []
    current_chunk = []
    current_entities = []  # entités du dernier paragraphe seulement
    current_length = 0
    current_centroid = None
    last_emb = None
    n_in_chunk = 0

    embeddings = model.encode(paragraphs, convert_to_tensor=True)

    for i, p_i in enumerate(paragraphs):
        doc = nlp(p_i)
        l_i = len(p_i.split())  # approximation tokens plus cohérente avec l_max
        entities_i = get_entities(doc)
        emb_i = embeddings[i]

        if current_centroid is None:
            s_i, e_i = 1.0, 1.0
        else:
            s_centroid = util.cos_sim(emb_i, current_centroid).item()
            s_last = util.cos_sim(emb_i, last_emb).item()
            s_i = 0.6 * s_centroid + 0.4 * s_last
            e_i = jaccard_weighted(entities_i, current_entities)  # vs dernier §

        if current_chunk and (
                s_i < theta_s or e_i < theta_e or current_length + l_i > l_max
        ):
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_entities, current_length = [], [], 0
            current_centroid, last_emb, n_in_chunk = None, None, 0

        current_chunk.append(p_i)
        current_entities = entities_i  # fenêtre = dernier paragraphe
        current_length += l_i
        last_emb = emb_i
        n_in_chunk += 1

        # Centroïde exact
        if current_centroid is None:
            current_centroid = emb_i.clone()
        else:
            current_centroid = (current_centroid * (n_in_chunk - 1) + emb_i) / n_in_chunk

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks