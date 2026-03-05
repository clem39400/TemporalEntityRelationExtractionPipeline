import os
import re
import unicodedata
from collections import Counter
from docling.document_converter import DocumentConverter
from MitreWhitelistLoader import MitreWhitelistLoader

class CTITextCleaner:
    def __init__(self, whitelist_ttl_days: int = 7):
        # Chargement de la whitelist dynamique
        self.cti_whitelist: set[str] = MitreWhitelistLoader(
            ttl_days=whitelist_ttl_days
        ).get_whitelist()

        # Initialisation du convertisseur IBM Docling
        self.converter = DocumentConverter()

    # ── 1. Extraction (Propulsée par Docling) ─────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """
        Extrait le texte structuré en Markdown via IBM Docling.
        Gère nativement les tableaux, listes, et ignore les en-têtes/pieds de page.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        try:
            # Conversion du document (PDF, Word, etc.)
            result = self.converter.convert(file_path)
            # Export direct en Markdown pour préserver la structure (titres, tableaux)
            return result.document.export_to_markdown()
        except Exception as e:
            raise RuntimeError(f"Échec de la conversion Docling pour {file_path}: {str(e)}")

    # ── 2. Normalisation ──────────────────────────────────────────────────────

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text

    # ── 3. Isolation et Filtrage ──────────────────────────────────────────────

    def clean_boilerplate(self, text: str) -> str:
        # On garde tes filtres CTI spécifiques, mais on enlève ceux liés
        # aux numéros de page car Docling s'en charge généralement.
        text = re.sub(r'(?i)May Cyber Threat Intelligence monthly report.*?\d{4}-\d{2}-\d{2}', '', text)
        text = re.sub(r'(?i)CERT aDvens\s*-\s*CTI\s*Advens.*?(?:Paris|\[\])', '', text)
        text = re.sub(r'(?i)\bTLP:\s*(RED|AMBER(?:[-+]\w+)?|GREEN|CLEAR|WHITE)\b', '', text)

        text = re.sub(r'(?i)^\s*(?:Table\s+of\s+contents?|Sommaire)\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?i)all rights reserved.*?\.', '', text)
        text = re.sub(r'(?i)©\s*\d{4}.*?\.', '', text)

        # Suppression du bruit très répétitif
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip())
        repeated_noise = {l for l, c in line_counts.items() if c > 3 and len(l) < 250}
        cleaned_lines = [l for l in lines if l.strip() not in repeated_noise]

        result = '\n'.join(cleaned_lines)
        return re.sub(r' +', ' ', result)

    def separate_ioc_block(self, text: str) -> tuple[str, str]:
        # La regex reste identique, elle marchera encore mieux sur du Markdown propre
        ioc_block = ""
        match = re.search(
            r'((?:Indicators?\s+of\s+Compromise|IoC[s]?|INDICATORS|NETWORK ARTIFACTS|HOST ARTIFACTS)\b.*)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            ioc_block = match.group(1).strip()
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

        # Nettoyage final des sauts de ligne multiples
        final_prose = re.sub(r'\n{3,}', '\n\n', final_prose).strip()

        return final_prose, ioc_block