"""
CTITextCleaner.py
-----------------
Module de préparation et d'assainissement des rapports CTI (PDF/TXT).
Assure l'extraction, la normalisation, la séparation des blocs IoC, 
la sanitisation (OpSec) et l'anonymisation PII, tout en préservant 
l'intégrité sémantique et structurelle du texte narratif.
"""

import os
import re
import unicodedata
import fitz  # PyMuPDF
from collections import Counter
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from MitreWhitelistLoader import MitreWhitelistLoader

class CTITextCleaner:
    def __init__(self, whitelist_ttl_days: int = 7):
        # Chargement unique des modèles NLP lourds
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Chargement de la whitelist dynamique
        self.cti_whitelist: set[str] = MitreWhitelistLoader(
            ttl_days=whitelist_ttl_days
        ).get_whitelist()

    # ── 1. Extraction ─────────────────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """
        Extrait le texte brut avec un filtrage spatial (cropping) pour 
        ignorer physiquement les en-têtes et pieds de page.
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
                        
                        # 2. Extraire uniquement à l'intérieur de cette zone
                        blocks = page.get_text("blocks", clip=clip_rect)
                        
                        text_blocks = [b for b in blocks if b[6] == 0]
                        text_blocks.sort(key=lambda b: (round(b[1] / 15), b[0]))
                        
                        current_line_y = None
                        current_line_text = []
                        
                        for b in text_blocks:
                            line_y = round(b[1] / 15)
                            text = re.sub(r'\s+', ' ', b[4].strip())
                            
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
        """
        Applique la forme canonique NFC, supprime les caractères de contrôle,
        et résout les césures typographiques.
        """
        # Forme canonique NFC pour préserver les hashes
        text = unicodedata.normalize('NFC', text)
        
        # Suppression des caractères de contrôle (garde \n et \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Résolution des césures typographiques (ex: "cam-\npaigns" -> "campaigns")
        text = re.sub(r'-\n(\S)', r'\1', text)
        
        return text

    def stitch_broken_sentences(self, text: str) -> str:
        """
        Reconnecte les phrases coupées artificiellement par des sauts de page 
        ou des blocs de boilerplate supprimés.
        """
        # Si une ligne se termine sans ponctuation terminale (ni point, exclamation, 
        # interrogation, deux-points) et que la reprise commence par une minuscule, 
        # on fusionne l'ensemble en supprimant les sauts de ligne intermédiaires.
        text = re.sub(r'(?<![.!?:\x22\x27])\s*\n+\s*([a-z])', r' \1', text)
        
        # Collapse des espaces multiples générés par les fusions
        text = re.sub(r' +', ' ', text)
        return text

    # ── 3. Isolation et Filtrage ──────────────────────────────────────────────

    def clean_boilerplate(self, text: str) -> str:
        """
        Supprime les métadonnées de mise en page, les headers/footers répétitifs 
        et les mentions légales qui parasitent l'analyse sémantique.
        """
        # Mentions TLP (Traffic Light Protocol)
        text = re.sub(r'(?i)^\s*TLP:\s*(RED|AMBER(?:[-+]\w+)?|GREEN|CLEAR|WHITE)\s*$', '', text, flags=re.MULTILINE)
        
        # Numérotation de pages (ex: "Page 1", "1 / 15", ou chiffres isolés)
        text = re.sub(r'(?i)^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Disclaimers légaux et droits d'auteur
        text = re.sub(r'(?i)all rights reserved.*?\.', '', text)
        text = re.sub(r'(?i)©\s*\d{4}.*?\.', '', text)

        # Identification et suppression des headers/footers répétés (> 3 occurrences)
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip())
        
        # On ne filtre que les lignes courtes pour ne pas supprimer par erreur 
        # une phrase légitime qui se répéterait.
        repeated_noise = {l for l, c in line_counts.items() if c > 3 and len(l) < 80}
        
        cleaned_lines = [l for l in lines if l.strip() not in repeated_noise]
        
        return '\n'.join(cleaned_lines)

    def separate_ioc_block(self, text: str) -> tuple[str, str]:
        """
        Isole les tables d'indicateurs de compromission (IoC) généralement 
        situées en fin de rapport, pour éviter de polluer le chunking sémantique.
        """
        ioc_block = ""
        # Recherche du titre de la section IoC (tolérant sur la casse et les pluriels)
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
        """Defang les adresses email et URLs narratives pour la sécurité opérationnelle."""
        # hacker@domain.com -> hacker[at]domain[.]com
        text = re.sub(
            r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
            lambda m: f"{m.group(1)}[at]{m.group(2).replace('.', '[.]')}",
            text
        )
        # http://malicious.com/payload -> hxxp://malicious[.]com/payload
        text = re.sub(
            r'https?://([a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;=]+)',
            lambda m: f"hxxp://{m.group(1).replace('.', '[.]')}",
            text
        )
        return text

    def anonymize_data(self, text: str) -> str:
        """
        Masque les PII (emails, téléphones) tout en préservant les entités
        techniques et les acteurs de la menace grâce à la whitelist CTI.
        """
        results = self.analyzer.analyze(
            text=text,
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
            language='en'
        )
        
        # On exclut les correspondances qui font partie du lexique CTI (ex: APT teams)
        safe_results = [
            r for r in results
            if text[r.start:r.end].lower() not in self.cti_whitelist
        ]
        
        if not safe_results:
            return text
            
        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=safe_results
        ).text

    # ── 5. Point d'Entrée ─────────────────────────────────────────────────────

    def process_file(self, file_path: str) -> tuple[str, str]:
        """
        Exécute la pipeline de nettoyage séquentielle sur un fichier source.
        
        Retourne :
            - prose_clean (str) : Récit narratif nettoyé, reconstitué et prêt pour le chunking.
            - ioc_block (str)   : Bloc de données techniques brutes extrait.
        """
        raw = self.extract_text(file_path)
        if not raw:
            return "", ""

        # 1. Normalisation de base
        normalized = self.normalize_text(raw)
        
        # 2. Séparation structurelle (Prose vs Données tabulaires)
        prose, ioc_block = self.separate_ioc_block(normalized)
        
        # 3. Nettoyage du bruit documentaire
        cleaned_prose = self.clean_boilerplate(prose)
        
        # 4. Reconstruction sémantique (résout le bug des phrases coupées)
        stitched_prose = self.stitch_broken_sentences(cleaned_prose)
        
        # 5. Assainissement et anonymisation
        sanitized = self.sanitize_iocs(stitched_prose)
        final_prose = self.anonymize_data(sanitized)

        # Nettoyage final des doubles sauts de ligne excessifs
        final_prose = re.sub(r'\n{3,}', '\n\n', final_prose).strip()

        return final_prose, ioc_block