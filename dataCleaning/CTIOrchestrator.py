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

Usage CLI :
    # Fichier unique
    python CTIOrchestrator.py --input report.pdf

    # Dossier entier avec sauvegarde JSON
    python CTIOrchestrator.py --input ./reports --output ./chunks_output

    # Seuils personnalisés
    python CTIOrchestrator.py --input report.pdf --theta_s 0.3 --theta_e 0.1 --l_max 600
"""

import os
import re
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Optional

# ── Imports internes ──────────────────────────────────────────────────────────
from CTITextCleaner import CTITextCleaner
from SemanticChunk import semantic_chunking_improved

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
    theta_e: float = 0.15    # 0.0 = désactivé (Jaccard < 0 impossible)
    l_max: int = 400        # longueur max d'un chunk en mots

@dataclass
class ProcessedDocument:
    """Résultat du pipeline pour un fichier traité."""
    source_file: str
    prose_clean: str
    ioc_block: str          # conservé pour transmission au LLM, non traité par regex
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

# Blocs PDF < MIN_BLOCK_WORDS mots sont fusionnés avec leur voisin
# pour garantir des embeddings fiables dans le chunker sémantique.
MIN_BLOCK_WORDS = 30

def split_into_paragraphs(text: str) -> list[str]:
    """
    Stage 1 du chunking.

    1. Découpe sur \\n\\n et nettoie les \\n résiduels intra-bloc.
    2. Fusionne les blocs trop courts (< MIN_BLOCK_WORDS mots) avec leur voisin
       suivant — évite les micro-blocs dont l'embedding serait peu fiable.
    3. Filtre les blocs vides ou inférieurs à 20 caractères.
    """
    # ── Étape 1 : nettoyage de base ───────────────────────────────────────────
    raw = [p.strip() for p in text.split("\n\n")]
    raw = [re.sub(r'\n+', ' ', p) for p in raw]
    raw = [re.sub(r' +', ' ', p) for p in raw]
    raw = [p for p in raw if len(p) >= 20]

    # ── Étape 2 : fusion des blocs trop courts ────────────────────────────────
    merged: list[str] = []
    buffer = ""
    for block in raw:
        if buffer:
            block = buffer + " " + block
            buffer = ""
        if len(block.split()) < MIN_BLOCK_WORDS:
            buffer = block   # accumuler avec le suivant
        else:
            merged.append(block)
    if buffer:               # dernier bloc résiduel trop court
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    return merged


def save_results(doc: ProcessedDocument, output_dir: str) -> None:
    """Sauvegarde les chunks en JSON (un fichier par document source)."""
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

    log.info(f"  → Sauvegardé : {out_path}")


# ── Traitement d'un fichier ───────────────────────────────────────────────────

def process_single_file(
    file_path: str,
    cleaner: CTITextCleaner,
    chunking_cfg: ChunkingConfig,
    output_dir: Optional[str] = None,
) -> ProcessedDocument:
    """
    Orchestre le pipeline complet pour un fichier :

        Étape 1 — CTITextCleaner.process_file()
                  extraction → normalisation → séparation IoC/prose
                  → sanitisation IoCs → anonymisation PII → boilerplate

        Étape 2 — split_into_paragraphs()
                  Stage 1 chunking : découpage structurel + fusion blocs courts

        Étape 3 — semantic_chunking_improved()
                  Stage 2 chunking : raffinement sémantique

        Étape 4 — save_results() [optionnel]
                  Sérialisation JSON si --output fourni

    L'extraction des entités et relations (NER + RE) est assurée en aval par le LLM.
    """
    log.info(f"Traitement : {file_path}")

    # ── Étape 1 : Nettoyage ───────────────────────────────────────────────────
    try:
        prose_clean, ioc_block = cleaner.process_file(file_path)
    except Exception as e:
        log.error(f"  Erreur nettoyage : {e}")
        return ProcessedDocument(
            source_file=file_path, prose_clean="", ioc_block="",
            paragraphs=[], chunks=[], error=str(e)
        )

    if not prose_clean:
        log.warning(f"  Texte vide après nettoyage : {file_path}")
        return ProcessedDocument(
            source_file=file_path, prose_clean="", ioc_block="",
            paragraphs=[], chunks=[], error="empty_after_cleaning"
        )

    # ── Étape 2 : Paragraphes — Stage 1 ──────────────────────────────────────
    paragraphs = split_into_paragraphs(prose_clean)
    log.info(f"  {len(paragraphs)} paragraphes extraits")

    if not paragraphs:
        log.warning(f"  Aucun paragraphe détecté : {file_path}")
        return ProcessedDocument(
            source_file=file_path, prose_clean=prose_clean, ioc_block=ioc_block,
            paragraphs=[], chunks=[], error="no_paragraphs"
        )

    # ── Étape 3 : Chunking sémantique — Stage 2 ──────────────────────────────
    try:
        chunks = semantic_chunking_improved(
            paragraphs=paragraphs,
            theta_s=chunking_cfg.theta_s,
            theta_e=chunking_cfg.theta_e,
            l_max=chunking_cfg.l_max,
        )
        log.info(
            f"  {len(chunks)} chunks produits "
            f"(θs={chunking_cfg.theta_s}, θe={chunking_cfg.theta_e}, "
            f"L_max={chunking_cfg.l_max})"
        )
    except Exception as e:
        log.error(f"  Erreur chunking : {e}")
        return ProcessedDocument(
            source_file=file_path, prose_clean=prose_clean, ioc_block=ioc_block,
            paragraphs=paragraphs, chunks=[], error=str(e)
        )

    # ── Construction du résultat ──────────────────────────────────────────────
    doc = ProcessedDocument(
        source_file=file_path,
        prose_clean=prose_clean,
        ioc_block=ioc_block,
        paragraphs=paragraphs,
        chunks=chunks,
    )

    # ── Étape 4 : Sauvegarde (optionnelle) ───────────────────────────────────
    if output_dir:
        save_results(doc, output_dir)

    log.info(f"  ✓ {doc.summary()}")
    return doc


# ── Traitement en lot ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def process_directory(
    input_dir: str,
    cleaner: CTITextCleaner,
    chunking_cfg: ChunkingConfig,
    output_dir: Optional[str] = None,
) -> list[ProcessedDocument]:
    """Applique process_single_file à tous les fichiers supportés du dossier."""
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        log.warning(f"Aucun fichier .pdf/.txt trouvé dans : {input_dir}")
        return []

    log.info(f"{len(files)} fichier(s) détecté(s) dans {input_dir}")
    results = []
    for fp in sorted(files):
        result = process_single_file(fp, cleaner, chunking_cfg, output_dir)
        results.append(result)

    ok = [r for r in results if not r.error]
    total_chunks = sum(len(r.chunks) for r in ok)
    n_errors = sum(1 for r in results if r.error)
    log.info(
        f"\n{'─' * 50}\n"
        f"Résumé : {len(ok)}/{len(results)} fichiers OK | "
        f"{total_chunks} chunks au total | "
        f"{n_errors} erreurs"
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    _defaults = ChunkingConfig()   # source de vérité unique
    parser = argparse.ArgumentParser(
        description="Pipeline CTI : nettoyage + chunking sémantique"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Fichier unique (.pdf/.txt) ou dossier contenant des rapports CTI"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Dossier de sortie pour les JSON (optionnel)"
    )
    parser.add_argument(
        "--theta_s", type=float, default=_defaults.theta_s,
        help=f"Seuil similarité sémantique (défaut : {_defaults.theta_s})"
    )
    parser.add_argument(
        "--theta_e", type=float, default=_defaults.theta_e,
        help=f"Seuil overlap entités Jaccard (défaut : {_defaults.theta_e})"
    )
    parser.add_argument(
        "--l_max", type=int, default=_defaults.l_max,
        help=f"Longueur max d'un chunk en mots (défaut : {_defaults.l_max})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("Initialisation du pipeline de nettoyage (Presidio + spaCy)...")
    cleaner = CTITextCleaner()

    cfg = ChunkingConfig(
        theta_s=args.theta_s,
        theta_e=args.theta_e,
        l_max=args.l_max,
    )

    input_path = args.input

    if os.path.isfile(input_path):
        result = process_single_file(input_path, cleaner, cfg, args.output)

        print(f"\n{'═' * 60}")
        print(f"Fichier : {result.source_file}")
        print(f"{'═' * 60}")
        for i, chunk in enumerate(result.chunks, 1):
            preview = chunk[:120].replace("\n", " ")
            print(f"\n[Chunk {i:02d}] {preview}...")

    elif os.path.isdir(input_path):
        process_directory(input_path, cleaner, cfg, args.output)

    else:
        log.error(f"Chemin invalide : {input_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()