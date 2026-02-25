import os
import re
import unicodedata
import fitz
from collections import Counter
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class CTIDataPipeline:

    # Whitelist MITRE ATT&CK + security vendors — à enrichir selon le corpus
    CTI_ENTITY_WHITELIST = {
        "lazarus", "apt29", "apt28", "apt1", "apt10", "apt40",
        "fancy bear", "cozy bear", "sandworm", "scattered spider",
        "crowdstrike", "fireeye", "mandiant", "symantec", "microsoft",
        "trend micro", "checkpoint", "paloalto", "recorded future",
    }

    def __init__(self):
        # Chargement unique des moteurs Presidio (coûteux, ne pas réinstancier en boucle)
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """Extrait le texte brut depuis un PDF ou un fichier TXT."""
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext.lower() == '.pdf':
            text = ""
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text("text") + "\n"
            except Exception as e:
                print(f"[extract_text] Erreur lecture PDF {file_path}: {e}")
            return text
        raise ValueError(f"Type de fichier non supporté : {ext}")

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize_text(self, text: str) -> str:
        """Normalise l'encodage et supprime les artefacts de mise en page."""
        # NFC : composition canonique (préserve les hashes et chemins de fichiers)
        text = unicodedata.normalize('NFC', text)
        # Suppression des caractères de contrôle non imprimables (hors \t et \n)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Collapse des espaces multiples
        text = re.sub(r' +', ' ', text)
        return text

    # ── Séparation prose / IoC ────────────────────────────────────────────────

    def separate_ioc_block(self, text: str) -> tuple:
        """
        Détache le bloc IoC structuré (tables de hashes, IPs, domaines)
        de la prose narrative, conformément à LLM-TIKG (Hu et al., 2024),
        Algorithm 1 Line 5.

        Retourne : (prose, ioc_block)
        """
        ioc_block = ""
        match = re.search(
            r'((?:Indicators?\s+of\s+Compromise|IoC[s]?|Appendix|INDICATORS).*)',
            text, re.IGNORECASE | re.DOTALL
        )
        if match:
            ioc_block = match.group(1)
            text = text[:match.start()]
        return text, ioc_block

    # ── Sanitisation des IoCs ─────────────────────────────────────────────────

    def sanitize_iocs(self, text: str) -> str:
        """
        Obfusque les IoCs pour la sécurité opérationnelle (évite l'exécution
        accidentelle de liens malveillants) sans les supprimer.
        Approche TiKG (Mouiche & Saad, 2025), Phase 1.
        """
        # Emails : hacker@domain.com → hacker[at]domain.com
        text = re.sub(
            r'([\w.\-]+)@([\w.\-]+\.\w+)',
            lambda m: f"{m.group(1)}[at]{m.group(2)}",
            text
        )
        # URLs malveillantes : http://mal.com → hxxp://mal[.]com
        text = re.sub(
            r'https?://([\w./\-%?=&]+)',
            lambda m: f"hxxp://{m.group(1).replace('.', '[.]', 1)}",
            text
        )
        return text

    # ── Anonymisation PII ─────────────────────────────────────────────────────

    def anonymize_data(self, text: str) -> str:
        """
        Anonymise le PII non-CTI (emails, numéros de téléphone).
        PERSON intentionnellement exclu pour préserver les threat actors.
        Les entités de la whitelist CTI ne sont pas masquées.
        """
        results = self.analyzer.analyze(
            text=text,
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
            language='en'
        )
        # Filtrage : ne pas masquer les entités CTI connues
        safe_results = [
            r for r in results
            if text[r.start:r.end].lower() not in self.CTI_ENTITY_WHITELIST
        ]
        if not safe_results:
            return text
        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=safe_results
        ).text

    # ── Nettoyage boilerplate ─────────────────────────────────────────────────

    def clean_boilerplate(self, text: str) -> str:
        """
        Supprime les éléments récurrents non informatifs :
        TLP headers, numéros de page isolés, lignes répétées (headers/footers PDF).
        """
        # TLP Traffic Light Protocol headers
        text = re.sub(r'(?i)TLP:\s*(RED|AMBER|GREEN|CLEAR|WHITE)', '', text)
        # Numéros de page isolés sur une ligne
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Disclaimers légaux fréquents
        text = re.sub(r'(?i)all rights reserved.*?\.', '', text)
        # Lignes répétées > 3 fois (headers/footers extraits par PyMuPDF)
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip())
        repeated = {l for l, c in line_counts.items() if c > 3 and len(l) < 100}
        lines = [l for l in lines if l.strip() not in repeated]
        return '\n'.join(lines)

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def process_file(self, file_path: str) -> tuple:
        """
        Exécute le pipeline complet sur un fichier.

        Retourne : (prose_nettoyée, ioc_block)
            - prose_nettoyée : texte normalisé, anonymisé, prêt pour le chunking
            - ioc_block      : bloc IoC structuré à traiter séparément par regex
        """
        raw = self.extract_text(file_path)
        if not raw:
            return "", ""

        normalized = self.normalize_text(raw)
        prose, ioc_block = self.separate_ioc_block(normalized)
        sanitized = self.sanitize_iocs(prose)
        anonymized = self.anonymize_data(sanitized)
        cleaned = self.clean_boilerplate(anonymized)

        return cleaned, ioc_block
