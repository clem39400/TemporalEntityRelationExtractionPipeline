"""
CTIOrchestrator.py
------------------
Point d'entrée principal du pipeline CTI.

Flux :
    fichier(s)
        → CTITextCleaner         : extraction, normalisation, séparation IoC/prose,
                                   sanitisation IoCs, anonymisation PII, boilerplate
        → split_into_paragraphs  : Stage 1 — découpage structurel (\n\n) + fusion des blocs courts
        → semantic_chunking_improved : Stage 2 — raffinement sémantique

    Le bloc IoC est séparé de la prose mais n'est PAS traité par regex :
    l'extraction des entités et relations sera assurée par le LLM en aval.
"""

import os
import json
import argparse
import re
import logging
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# ── Imports internes ──────────────────────────────────────────────────────────
from CTITextCleaner import CTITextCleaner
from SemanticChunk import semantic_chunking_improved
from LLMEngine import LLMEngine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Désactiver les warnings verbeux de Presidio
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    """Hyperparamètres de l'algorithme de chunking sémantique (SC-LKM)."""
    theta_s: float = 0.2    # seuil similarité cosinus
    theta_e: float = 0.15   # seuil overlap entités Jaccard
    l_max: int = 400        # longueur max d'un chunk en mots

@dataclass
class ProcessedDocument:
    """Résultat du pipeline pour un fichier traité."""
    source_file: str
    prose_clean: str
    ioc_block: str
    paragraphs: list[str]
    chunks: list[str]
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"[{os.path.basename(self.source_file)}] "
            f"{len(self.paragraphs)} paragraphes → "
            f"{len(self.chunks)} chunks"
        )


# ── Utilitaires ───────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str) -> list[str]:
    """Stage 1 du chunking (Découpage structurel + fusion des blocs courts)."""
    MIN_BLOCK_WORDS = 30
    raw = [p.strip() for p in text.split("\n\n")]
    raw = [p for p in raw if len(p) >= 10]

    merged: list[str] = []
    buffer = ""

    for block in raw:
        if buffer:
            block = buffer + "\n\n" + block
            buffer = ""

        # Gestion des citations
        prev_ends_with_quote = False
        if merged and merged[-1].strip():
            last_char = merged[-1].strip()[-1]
            if last_char in ("'", "\"", "”", "’", "»", "«"):
                prev_ends_with_quote = True

        if prev_ends_with_quote and len(block.split()) < 20:
            merged[-1] += "\n\n" + block
            continue

        if len(block.split()) < MIN_BLOCK_WORDS and not block.startswith(("#", "|", "-", "*")):
            buffer = block
        else:
            merged.append(block)

    if buffer:
        if merged:
            merged[-1] += "\n\n" + buffer
        else:
            merged.append(buffer)

    return merged

def is_semantic_content(text: str) -> bool:
    """
    Filtre avancé pour éliminer les sommaires, listes de contributeurs, et métadonnées.
    """
    text_clean = text.strip()
    words = text_clean.split()

    # 1. Filtre de taille (Trop court)
    # On conserve les titres Markdown qui donnent du contexte
    if len(words) < 10 and not text_clean.startswith(("#", "-", "*")):
        return False

    # 2. Rejet par mot-clé (En-têtes de sommaires)
    lower_text = text_clean.lower()
    if lower_text.startswith("table of contents") or lower_text.startswith("## contents") or lower_text.startswith("sommaire"):
        return False

    lines = [l.strip() for l in text_clean.split('\n') if l.strip()]

    # 3. Détection des Tables des Matières (TOC)
    # Cherche les lignes qui finissent par un numéro de page ex: "| Titre | 4 |" ou "Titre .... 4"
    toc_pattern = re.compile(r'(\|\s*\d+\s*\|?$|\.{3,}\s*\d+\s*\|?$)')
    toc_lines = sum(1 for l in lines if toc_pattern.search(l))

    # Si plus de 30% du bloc ressemble à un sommaire, on jette
    if len(lines) > 2 and (toc_lines / len(lines)) > 0.3:
        return False

        # 4. Détection des listes de noms / sponsors / adresses
    # Une vraie narration cyber contient de la ponctuation de fin de phrase (. ! ?)
    # Si on a un bloc de plus de 25 mots avec 1 seul (ou 0) point, c'est généralement une liste ou un copyright.
    sentences = [s for s in re.split(r'[.!?]', text_clean) if len(s.strip()) > 0]
    if len(words) > 25 and len(sentences) <= 1:
        return False

    # 5. Détection des blocs de pur copyright ou d'index
    if "all rights reserved" in lower_text or "issn:" in lower_text:
        return False

    return True

def save_results(doc: ProcessedDocument, output_dir: str) -> str:
    """Sauvegarde les chunks en JSON et retourne le chemin du fichier créé."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(doc.source_file))[0]
    out_path = os.path.join(output_dir, f"{base}_processed.json")

    payload = {
        "source_file": doc.source_file,
        "n_paragraphs": len(doc.paragraphs),
        "n_chunks": len(doc.chunks),
        "chunks": doc.chunks,
        "ioc_block": doc.ioc_block,
        "error": doc.error,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"  → Chunks sauvegardés : {out_path}")
    return out_path


# ── Traitement d'un fichier ───────────────────────────────────────────────────

def process_single_file(
        file_path: str,
        cleaner: CTITextCleaner,
        chunking_cfg: ChunkingConfig,
        output_dir: Optional[str] = None,
) -> ProcessedDocument:
    """Orchestre le nettoyage et le chunking sémantique pour un fichier."""
    log.info(f"Traitement : {file_path}")

    try:
        prose_clean, ioc_block = cleaner.process_file(file_path)
    except Exception as e:
        log.error(f"  Erreur nettoyage : {e}")
        return ProcessedDocument(file_path, "", "", [], [], error=str(e))

    # 1. Découpage brut
    paragraphs = split_into_paragraphs(prose_clean)

    # 2. NOUVEAU : Filtrage du bruit sémantique AVANT le chunking
    paragraphs_filtered = [p for p in paragraphs if is_semantic_content(p)]
    log.info(f"  Paragraphes filtrés : {len(paragraphs)} -> {len(paragraphs_filtered)} utiles")

    try:
        chunks = semantic_chunking_improved(
            paragraphs=paragraphs_filtered, # On utilise la liste nettoyée
            theta_s=chunking_cfg.theta_s,
            theta_e=chunking_cfg.theta_e,
            l_max=chunking_cfg.l_max,
        )
    except Exception as e:
        log.error(f"  Erreur chunking : {e}")
        return ProcessedDocument(file_path, prose_clean, ioc_block, paragraphs_filtered, [], error=str(e))

    doc = ProcessedDocument(file_path, prose_clean, ioc_block, paragraphs_filtered, chunks)

    if output_dir:
        save_results(doc, output_dir)

    log.info(f"  ✓ {doc.summary()}")
    return doc


# ── Traitement en lot ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

def process_directory(input_dir, cleaner, chunking_cfg, output_dir, llm_engine):
    """Parcourt un dossier et lance le traitement LLM en parallèle via Threads."""
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        log.warning(f"Aucun fichier supporté trouvé dans : {input_dir}")
        return []

    log.info(f"{len(files)} fichier(s) détecté(s). Lancement du pipeline...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        for fp in sorted(files):
            doc = process_single_file(fp, cleaner, chunking_cfg, output_dir)

            if not doc.error and output_dir and llm_engine:
                base = os.path.splitext(os.path.basename(doc.source_file))[0]
                json_path = os.path.join(output_dir, f"{base}_processed.json")
                # Envoi asynchrone au LLM
                executor.submit(llm_engine.process_json_file, json_path)


# ── CLI & Main ────────────────────────────────────────────────────────────────

def parse_args():
    _defaults = ChunkingConfig()
    parser = argparse.ArgumentParser(description="Pipeline CTI : Nettoyage + Chunking + LLM")
    parser.add_argument("--input", "-i", required=True, help="Fichier ou dossier d'entrée")
    parser.add_argument("--output", "-o", required=True, help="Dossier pour les chunks JSON")
    parser.add_argument("--llm_output", default="./relations", help="Dossier pour les relations extraites")
    parser.add_argument("--rocade", required=True, help="Chemin vers l'ontologie ROCADE JSON")
    parser.add_argument("--theta_s", type=float, default=_defaults.theta_s)
    parser.add_argument("--theta_e", type=float, default=_defaults.theta_e)
    parser.add_argument("--l_max", type=int, default=_defaults.l_max)
    return parser.parse_args()


def main():
    args = parse_args()
    log.info("Démarrage du pipeline CTI...")

    cleaner = CTITextCleaner()
    llm_engine = LLMEngine(output_dir=args.llm_output, rocade_json_path=args.rocade)
    cfg = ChunkingConfig(theta_s=args.theta_s, theta_e=args.theta_e, l_max=args.l_max)

    if os.path.isfile(args.input):
        log.info("Mode fichier unique.")
        doc = process_single_file(args.input, cleaner, cfg, args.output)
        if not doc.error:
            base = os.path.splitext(os.path.basename(doc.source_file))[0]
            json_path = os.path.join(args.output, f"{base}_processed.json")
            llm_engine.process_json_file(json_path)

    elif os.path.isdir(args.input):
        log.info(f"Mode répertoire : {args.input}")
        process_directory(args.input, cleaner, cfg, args.output, llm_engine)
    else:
        log.error(f"Entrée invalide : {args.input}")

if __name__ == "__main__":
    main()