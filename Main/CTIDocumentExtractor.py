import fitz  # PyMuPDF
import re
from unidecode import unidecode

class CTIDocumentExtractor:
    def __init__(self, remove_non_ascii: bool = True, hide_sensitive: bool = False, margin_tolerance: float = 0.08):
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
        """Extrait le texte du PDF en filtrant les marges."""
        valid_text_blocks = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_height = page.rect.height
                header_limit = page_height * self.margin_tolerance
                footer_limit = page_height * (1 - self.margin_tolerance)

                blocks = page.get_text("blocks")
                # Tri des blocs de haut en bas, puis de gauche à droite
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
        """Applique les règles de nettoyage de la Phase 1 avec filtres anti-bruit avancés."""
        if not text:
            return ""

        # 1. Suppression des lignes de sommaire (ex: "Hardware-Enabled Defense..........................8")
        text = re.sub(r'\.{4,}\s*\d+', ' ', text)

        # 2. Réduction des mots répétés en boucle à une seule occurrence (ex: "FOUNDER FOUNDER FOUNDER" -> "FOUNDER")
        text = re.sub(r'\b([A-Za-z]+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

        # 3. Masquage des données sensibles
        if self.hide_sensitive:
            text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[REDACTED_IP]', text)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[REDACTED_EMAIL]', text)

        # 4. Suppression des caractères non-ASCII
        if self.remove_non_ascii:
            text = unidecode(text)

        # 5. Lissage des espaces et sauts de ligne
        return re.sub(r'\s+', ' ', text).strip()