"""
CTITextCleaner.py
-----------------
Module de préparation et d'assainissement des rapports CTI (PDF/TXT).
Assure l'extraction, la normalisation, la séparation des blocs IoC, 
la sanitisation (OpSec) et l'anonymisation (Emails) ultra-rapide par Regex.
"""

import os
import re
import unicodedata
import fitz  # PyMuPDF
from collections import Counter
from MitreWhitelistLoader import MitreWhitelistLoader

class CTITextCleaner:
    def __init__(self, whitelist_ttl_days: int = 7):
        # Chargement de la whitelist dynamique
        self.cti_whitelist: set[str] = MitreWhitelistLoader(
            ttl_days=whitelist_ttl_days
        ).get_whitelist()

    # ── 1. Extraction ─────────────────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """
        Extrait le texte brut avec un filtrage spatial (cropping) pour 
        ignorer physiquement les en-têtes et pieds de page.
        L'extraction respecte l'ordre de lecture naturel (gère les colonnes).
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        elif ext.lower() == '.pdf':
            extracted_text = []
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        # 1. Définir la zone de lecture (ignorer top 50px et bottom 50px)
                        rect = page.rect
                        clip_rect = fitz.Rect(rect.x0, rect.y0 + 50, rect.x1, rect.y1 - 50)
                        
                        # 2. Extraire uniquement à l'intérieur de cette zone (tri naturel activé)
                        blocks = page.get_text("blocks", clip=clip_rect, sort=True)
                        
                        # On filtre pour ne garder que le texte (type 0)
                        text_blocks = [b for b in blocks if b[6] == 0]
                        
                        current_line_y = None
                        current_line_text = []
                        
                        for b in text_blocks:
                            raw_text = b[4].strip()
                            
                            # --- Filtres anti-Sommaires à la source ---
                            if re.search(r'(?:\.\s*){8,}', raw_text):
                                continue
                            if re.search(r'^(?:(?:[IVX]+|[A-Z])\s*)?→\s+.*\d{1,3}$', raw_text, flags=re.MULTILINE):
                                continue
                            if re.search(r'^\d+/\s+.*\d{1,3}$', raw_text, flags=re.MULTILINE):
                                continue
                            # ------------------------------------------
                            
                            line_y = round(b[1] / 15)
                            text = re.sub(r'\s+', ' ', raw_text)
                            
                            if not text:
                                continue
                                
                            if current_line_y is None or line_y == current_line_y:
                                current_line_text.append(text)
                                current_line_y = line_y
                            else:
                                extracted_text.append(" | ".join(current_line_text))
                                current_line_text = [text]
                                current_line_y = line_y
                                
                        if current_line_text:
                            extracted_text.append(" | ".join(current_line_text))
                            
                return "\n\n".join(extracted_text)
                
            except Exception as e:
                raise RuntimeError(f"Échec de la lecture du PDF {file_path}: {str(e)}")
                
        raise ValueError(f"Format de fichier non supporté : {ext}")

    # ── 2. Normalisation & Reconstruction ─────────────────────────────────────

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # On fusionne complètement le mot coupé (ex: "cybersecu- rity" -> "cybersecurity")
        text = re.sub(r'(?<=[a-z])-\s+([a-z])', r'\1', text)
        text = re.sub(r'(?<=\b[A-Z])\s(?=[A-Z]\b)', '', text)
        return text

    def stitch_broken_sentences(self, text: str) -> str:
        text = re.sub(r'([a-z]{2,}\.)([A-Z])', r'\1 \2', text)
        text = re.sub(r'(?<![.!?:\x22\x27])\s*\n+\s*([a-z])', r' \1', text)
        return re.sub(r' +', ' ', text)

    # ── 3. Isolation et Filtrage ──────────────────────────────────────────────

    def clean_boilerplate(self, text: str) -> str:
        text = re.sub(r'(?i)May Cyber Threat Intelligence monthly report.*?\d{4}-\d{2}-\d{2}', '', text)
        text = re.sub(r'(?i)CERT aDvens\s*-\s*CTI\s*Advens.*?(?:Paris|\[\])', '', text)
        text = text.replace('|', ' ')
        text = re.sub(r'(?i)\bTLP:\s*(RED|AMBER(?:[-+]\w+)?|GREEN|CLEAR|WHITE)\b', '', text)
        text = re.sub(r'(?i)/home/[\w/.-]+(?:{[\w]+})?\.png', '', text)
        
        text = re.sub(r'(?i)^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?i)^\s*(?:Table\s+of\s+contents?|Sommaire)\s*$', '', text, flags=re.MULTILINE)

        # Filtres spécifiques Sommaires ANSSI / CERT-FR (Le coup de balai final)
        text = re.sub(r'(?:→|\b\d+/)\s*[^.!?]{2,150}?\b\d{1,3}\b', '', text)
        text = re.sub(r'(?i)best\s+practices\s+for\s+information\s+system\s+security\s+\d{1,3}\b', '', text)
        text = re.sub(r'(?i)CYBER\s+THREAT\s+OVERVIEW(?:\s+\d{4})?', '', text)
        text = re.sub(r'(?i)ATTACK\s+OBJECTIVES|MEANS\s+EMPLOYED\s+BY\s+ATTACKERS', '', text)

        text = re.sub(r'(?i)all rights reserved.*?\.', '', text)
        text = re.sub(r'(?i)©\s*\d{4}.*?\.', '', text)

        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip())
        repeated_noise = {l for l, c in line_counts.items() if c > 3 and len(l) < 250}
        cleaned_lines = [l for l in lines if l.strip() not in repeated_noise]
        
        result = '\n'.join(cleaned_lines)
        return re.sub(r' +', ' ', result)

    def separate_ioc_block(self, text: str) -> tuple[str, str]:
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
        """
        NOUVEAU : Masque les emails 1000x plus rapidement via Regex, 
        tout en préservant la whitelist.
        """
        def replace_if_not_whitelisted(match):
            email = match.group(0)
            if email.lower() in self.cti_whitelist:
                return email
            return "[REDACTED_EMAIL]"

        # Match une adresse email standard
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', replace_if_not_whitelisted, text)

    # ── 5. Point d'Entrée ─────────────────────────────────────────────────────

    def process_file(self, file_path: str) -> tuple[str, str]:
        raw = self.extract_text(file_path)
        if not raw:
            return "", ""

        normalized = self.normalize_text(raw)
        prose, ioc_block = self.separate_ioc_block(normalized)
        cleaned_prose = self.clean_boilerplate(prose)
        stitched_prose = self.stitch_broken_sentences(cleaned_prose)
        
        # Inversion logique : on anonymise l'email AVANT de faire le defanging (pour que la regex match)
        anonymized_prose = self.anonymize_data(stitched_prose)
        final_prose = self.sanitize_iocs(anonymized_prose)

        final_prose = re.sub(r'\n{3,}', '\n\n', final_prose).strip()

        return final_prose, ioc_block