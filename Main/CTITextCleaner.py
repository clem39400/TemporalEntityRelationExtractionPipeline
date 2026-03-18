import os
import re
import unicodedata
import fitz  # PyMuPDF : Ultra rapide, aucune IA locale, 0 crash RAM
from collections import Counter
from MitreWhitelistLoader import MitreWhitelistLoader

class CTITextCleaner:
    def __init__(self, whitelist_ttl_days: int = 7):
        self.cti_whitelist: set[str] = MitreWhitelistLoader(
            ttl_days=whitelist_ttl_days
        ).get_whitelist()

    # ── 1. Extraction (100% PyMuPDF) ──────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """
        Extraction robuste et instantanée avec PyMuPDF.
        Utilise la détection de blocs pour reconstituer de vrais paragraphes.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        text_paragraphs = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    # 'blocks' extrait intelligemment les vrais paragraphes du PDF
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        # b[4] contient le texte brut du bloc
                        text = b[4].strip()

                        if text:
                            # On remplace les sauts de ligne "visuels" (au milieu d'une phrase) par des espaces
                            text = text.replace('\n', ' ')

                            # Petit filtre pour ignorer les artefacts (numéros de page isolés, etc.)
                            if len(text) > 15:
                                text_paragraphs.append(text)

            # On rejoint les vrais paragraphes avec un double saut de ligne
            # pour que SemanticChunk.py puisse les séparer correctement.
            return "\n\n".join(text_paragraphs)

        except Exception as e:
            raise RuntimeError(f"Échec total de l'extraction PyMuPDF pour {file_path}: {e}")

    # ── 2. Normalisation ──────────────────────────────────────────────────────

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text

    # ── 3. Isolation et Filtrage ──────────────────────────────────────────────

    def clean_boilerplate(self, text: str) -> str:
        text = re.sub(r'(?i)May Cyber Threat Intelligence monthly report.*?\d{4}-\d{2}-\d{2}', '', text)
        text = re.sub(r'(?i)CERT aDvens\s*-\s*CTI\s*Advens.*?(?:Paris|\[\])', '', text)
        text = re.sub(r'(?i)\bTLP:\s*(RED|AMBER(?:[-+]\w+)?|GREEN|CLEAR|WHITE)\b', '', text)
        text = re.sub(r'(?i)^\s*(?:Table\s+of\s+contents?|Sommaire)\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?i)all rights reserved.*?\.', '', text)
        text = re.sub(r'(?i)©\s*\d{4}.*?\.', '', text)

        # CORRECTION : On découpe par PARAGRAPHE (\n\n) et non par ligne
        paragraphs = text.split('\n\n')
        para_counts = Counter(p.strip() for p in paragraphs if p.strip())

        # Suppression du bruit (pieds de page répétés)
        repeated_noise = {p for p, c in para_counts.items() if c > 3 and len(p) < 250}
        cleaned_paras = [p for p in paragraphs if p.strip() not in repeated_noise]

        # On reconstruit avec de VRAIS séparateurs de paragraphes
        result = '\n\n'.join(cleaned_paras)
        return re.sub(r' +', ' ', result)

    def separate_ioc_block(self, text: str) -> tuple[str, str]:
        ioc_block = ""
        # CORRECTION : On exige un saut de ligne AVANT et APRES pour être sûr que c'est un TITRE
        # et pas juste une mention au milieu d'une phrase.
        match = re.search(
            r'\n\s*#*\s*((?:Indicators?\s+of\s+Compromise|IoC[s]?|INDICATORS|NETWORK ARTIFACTS|HOST ARTIFACTS)\b.*?)\n(.*)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            # On vérifie que ce n'est pas déclenché trop tôt (ex: Sommaire)
            # Un vrai bloc IoC se trouve généralement à la toute fin du rapport.
            if match.start() > (len(text) * 0.5):
                ioc_block = match.group(2).strip()
                text = text[:match.start()].strip()

        return text, ioc_block

    # ── 4. OpSec et Confidentialité ───────────────────────────────────────────

    def sanitize_iocs(self, text: str) -> str:
        text = re.sub(
            r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
            lambda m: f"{m.group(1)}[at]{m.group(2).replace('.', '[.]')}",
            text
        )
        text = re.sub(
            r'https?://([a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;=]+)',
            lambda m: f"hxxp://{m.group(1).replace('.', '[.]')}",
            text
        )
        return text

    def anonymize_data(self, text: str) -> str:
        def replace_if_not_whitelisted(match):
            email = match.group(0)
            if email.lower() in self.cti_whitelist:
                return email
            return "[REDACTED_EMAIL]"

        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', replace_if_not_whitelisted, text)

    # ── 5. Point d'Entrée ─────────────────────────────────────────────────────

    def process_file(self, file_path: str) -> tuple[str, str]:
        raw = self.extract_text(file_path)
        if not raw:
            return "", ""

        normalized = self.normalize_text(raw)
        prose, ioc_block = self.separate_ioc_block(normalized)
        cleaned_prose = self.clean_boilerplate(prose)

        anonymized_prose = self.anonymize_data(cleaned_prose)
        final_prose = self.sanitize_iocs(anonymized_prose)

        final_prose = re.sub(r'\n{3,}', '\n\n', final_prose).strip()

        return final_prose, ioc_block