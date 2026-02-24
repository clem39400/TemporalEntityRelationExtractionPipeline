import os
import re
import unicodedata
import fitz
from collections import Counter
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class CTIDataPipeline:

    # Whitelist MITRE ATT&CK à enrichir
    CTI_ENTITY_WHITELIST = {
        "lazarus", "apt29", "apt28", "apt1", "apt10", "apt40",
        "fancy bear", "cozy bear", "sandworm", "scattered spider",
        "crowdstrike", "fireeye", "mandiant", "symantec", "microsoft",
        "trend micro", "checkpoint", "paloalto", "recorded future",
    }

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def extract_text(self, file_path: str) -> str:
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
                print(f"Error reading PDF {file_path}: {e}")
            return text
        raise ValueError(f"Unsupported file type: {ext}")

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)  # NFC, pas NFKD
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r' +', ' ', text)
        return text

    def sanitize_iocs(self, text: str) -> str:
        text = re.sub(r'([\w.\-]+)@([\w.\-]+\.\w+)',
                      lambda m: f"{m.group(1)}[at]{m.group(2)}", text)
        text = re.sub(r'https?://([\w./\-%?=&]+)',
                      lambda m: f"hxxp://{m.group(1).replace('.', '[.]', 1)}", text)
        return text

    def separate_ioc_block(self, text: str) -> tuple:
        """Sépare le bloc IoC structuré de la prose."""
        ioc_block = ""
        match = re.search(
            r'((?:Indicators?\s+of\s+Compromise|IoC[s]?|Appendix|INDICATORS).*)',
            text, re.IGNORECASE | re.DOTALL
        )
        if match:
            ioc_block = match.group(1)
            text = text[:match.start()]
        return text, ioc_block

    def anonymize_data(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
            language='en'
        )
        safe_results = [
            r for r in results
            if text[r.start:r.end].lower() not in self.CTI_ENTITY_WHITELIST
        ]
        if not safe_results:
            return text
        return self.anonymizer.anonymize(text=text, analyzer_results=safe_results).text

    def clean_boilerplate(self, text: str) -> str:
        text = re.sub(r'(?i)TLP:\s*(RED|AMBER|GREEN|CLEAR|WHITE)', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?i)all rights reserved.*?\.', '', text)
        # Supprimer lignes répétées (headers/footers PDF)
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip())
        repeated = {l for l, c in line_counts.items() if c > 3 and len(l) < 100}
        lines = [l for l in lines if l.strip() not in repeated]
        return '\n'.join(lines)

    def process_file(self, file_path: str) -> tuple:
        raw = self.extract_tex