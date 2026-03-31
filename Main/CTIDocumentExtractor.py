import os
import fitz  # PyMuPDF
import re
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
from typing import List, Dict

class CTIDocumentExtractor:
    def __init__(self, remove_non_ascii: bool = True, hide_sensitive: bool = False, margin_tolerance: float = 0.08):
        """Extracteur hybride capable de lire des dossiers entiers de PDF et de TXT."""
        self.remove_non_ascii = remove_non_ascii
        self.hide_sensitive = hide_sensitive
        self.margin_tolerance = margin_tolerance

    def extract_file(self, file_path: str) -> str:
        """Route le fichier vers la bonne méthode d'extraction selon son extension."""
        ext = file_path.lower().split('.')[-1]

        if ext == 'pdf':
            raw_text = self._extract_pdf_heuristic(file_path)
        elif ext == 'txt':
            raw_text = self._extract_txt(file_path)
        else:
            print(f"Format non supporté ignoré : {file_path}")
            return ""

        return self._sanitize_text(raw_text)

    def _extract_txt(self, file_path: str) -> str:
        """Extrait le texte brut d'un fichier .txt."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Erreur de lecture TXT ({file_path}): {e}")
            return ""

    def _extract_pdf_heuristic(self, pdf_path: str) -> str:
        """Extrait le texte du PDF en filtrant géométriquement les marges."""
        valid_text_blocks = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_height = page.rect.height
                header_limit = page_height * self.margin_tolerance
                footer_limit = page_height * (1 - self.margin_tolerance)

                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))

                for b in blocks:
                    y0, y1, block_text, block_type = b[1], b[3], b[4], b[6]
                    if block_type == 0 and y0 > header_limit and y1 < footer_limit:
                        valid_text_blocks.append(block_text.strip())
            doc.close()
        except Exception as e:
            print(f"Erreur de lecture PDF ({pdf_path}): {e}")
            return ""

        return " ".join(valid_text_blocks)

    def _sanitize_text(self, text: str) -> str:
        """Applique les règles de nettoyage de la Phase 1."""
        if not text:
            return ""

        if self.hide_sensitive:
            text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[REDACTED_IP]', text)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[REDACTED_EMAIL]', text)

        if self.remove_non_ascii:
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        return re.sub(r'\s+', ' ', text).strip()