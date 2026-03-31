import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
from typing import List, Dict
import json
import os
from datetime import datetime

from Main.CTIDocumentExtractor import CTIDocumentExtractor


class CTISemanticChunker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.35, overlap_sentences: int = 1, device: str = 'cpu', batch_size: int = 32):
        """Initialise le module de chunking (Le modèle est chargé une seule fois ici)."""
        self.encoder = SentenceTransformer(model_name, device=device)
        self.similarity_threshold = similarity_threshold
        self.overlap = overlap_sentences
        self.batch_size = batch_size

    def _tokenize_sentences(self, text: str) -> List[str]:
        return nltk.sent_tokenize(" ".join(text.split()))

    def chunk_report(self, text: str, source_filename: str) -> List[Dict[str, any]]:
        """Segmente le texte et associe chaque chunk à son fichier source."""
        sentences = self._tokenize_sentences(text)

        if not sentences:
            return []
        if len(sentences) == 1:
            return [{"source": source_filename, "chunk_id": 0, "text": sentences[0], "sentence_count": 1}]

        embeddings = self.encoder.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        distances = paired_cosine_distances(embeddings[:-1], embeddings[1:])
        sim_scores = 1.0 - distances

        boundaries = np.where(sim_scores < self.similarity_threshold)[0] + 1
        boundaries = np.append(boundaries, len(sentences))

        chunks = []
        start_idx = 0

        for chunk_id, end_idx in enumerate(boundaries):
            chunk_sentences = sentences[start_idx:end_idx]
            chunks.append({
                "source": source_filename,
                "chunk_id": chunk_id,
                "text": " ".join(chunk_sentences),
                "sentence_count": len(chunk_sentences)
            })

            start_idx = max(0, end_idx - self.overlap)
            if start_idx >= end_idx:
                start_idx = end_idx

        return chunks


def run_batch_pipeline(directory_path: str):
    """Orchestre la Phase 1 pour un dossier entier."""
    print(f"--- Démarrage du traitement par lot sur : {directory_path} ---")

    # 1. Initialisation unique des classes
    extractor = CTIDocumentExtractor(remove_non_ascii=True, hide_sensitive=True, margin_tolerance=0.08)

    # Laissez device='cpu' pour l'instant pour éviter tout nouveau crash
    chunker = CTISemanticChunker(model_name='all-MiniLM-L6-v2', similarity_threshold=0.35, overlap_sentences=1, device='cpu')

    all_corpus_chunks = []

    # 2. Parcours du dossier
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Ignorer les sous-dossiers
        if not os.path.isfile(file_path):
            continue

        print(f"Traitement de : {filename}...")

        # Extraction
        clean_text = extractor.extract_file(file_path)
        if not clean_text:
            continue

        # Chunking
        file_chunks = chunker.chunk_report(clean_text, source_filename=filename)
        all_corpus_chunks.extend(file_chunks)

    print(f"\nPhase 1 terminée ! Total : {len(all_corpus_chunks)} chunks extraits depuis le dossier.")
    return all_corpus_chunks
def save_chunks_to_json(chunks_data: list, output_dir: str):
    """
    Sauvegarde la liste des chunks au format JSON dans le dossier spécifié.
    Crée un fichier horodaté pour éviter d'écraser les exécutions précédentes.
    """
    if not chunks_data:
        print("Aucun chunk à sauvegarder.")
        return None

    # Création du dossier cible s'il n'existe pas déjà
    os.makedirs(output_dir, exist_ok=True)

    # Génération d'un nom de fichier avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"corpus_chunks_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        # Encodage utf-8 et ensure_ascii=False pour préserver la lisibilité
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=4)

        print(f"\n[SUCCÈS] Les {len(chunks_data)} chunks ont été exportés dans :\n-> {filepath}")
        return filepath
    except Exception as e:
        print(f"\n[ERREUR] Impossible de sauvegarder le fichier JSON : {e}")
        return None

# --- Exécution principale mise à jour ---
if __name__ == "__main__":
    # 1. Définition des chemins
    # Chemin vers votre dataset source (dossier contenant vos TXT/PDF)
    dossier_source = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main\\InputData"

    # Chemin vers votre dossier de sortie JSON
    dossier_sortie = "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main\\OutputChunks"

    # (Optionnel) Création du dossier source si c'est la première exécution
    os.makedirs(dossier_source, exist_ok=True)

    # 2. Lancement de la Phase 1 (Extraction + Chunking par lot)
    # Assurez-vous d'avoir des fichiers dans le 'dossier_source' avant d'exécuter
    print(f"Lancement de la pipeline sur le dossier : {dossier_source}")
    resultats_batch = run_batch_pipeline(dossier_source)

    # 3. Exportation des résultats
    if resultats_batch:
        save_chunks_to_json(resultats_batch, dossier_sortie)