"""
CTIPipeline.py
--------------
Point d'entrée principal du pipeline CTI.

Flux :
    fichier(s)
        → CTIDataPipeline  : extraction, normalisation, sanitisation IoC,
                             anonymisation PII, nettoyage boilerplate
        → split_into_paragraphs : Stage 1 — découpage structurel (\n\n)
        → semantic_chunking_improved : Stage 2 — raffinement sémantique
        → process_ioc_block : extraction regex du bloc IoC (séparé du LLM)

Usage CLI :
    # Fichier unique
    python CTIPipeline.py --input report.pdf

    # Dossier entier avec sauvegarde JSON
    python CTIPipeline.py --input ./reports --output ./chunks_output

    # Seuils personnalisés
    python CTIPipeline.py --input report.pdf --theta_s 0.4 --theta_e 0.15 --l_max 350
"""

import os
import re
import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

# ── Imports internes ──────────────────────────────────────────────────────────
from CTIDataPipeline import CTIDataPipeline
from SemanticChunk import semantic_chunking_improved

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    """Hyperparamètres de l'algorithme de chunking sémantique (SC-LKM)."""
    theta_s: float = 0.5   # seuil similarité cosinus
    theta_e: float = 0.1   # seuil overlap entités (Jaccard)
    l_max: int = 400        # longueur max d'un chunk en mots


@dataclass
class ProcessedDocument:
    source_file: str
    prose_clean: str
    ioc_block: str
    paragraphs: list[str]
    chunks: list[str]
    ioc_relations: list[tuple] = field(default_factory=list)
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"[{os.path.basename(self.source_file)}] "
            f"{len(self.paragraphs)} paragraphes → "
            f"{len(self.chunks)} chunks | "
            f"{len(self.ioc_relations)} IoC relations"
        )


# ── Utilitaires ───────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str) -> list[str]:
    """
    Stage 1 du chunking : découpe structurelle par double saut de ligne.
    Filtre les paragraphes vides ou trop courts (< 20 caractères).
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if len(p) >= 20]


def process_ioc_block(ioc_block: str, source_file: str) -> list[tuple]:
    """
    Extrait les IoCs structurés depuis le bloc séparé et les retourne
    sous forme de triplets (source, relation, valeur).

    Implémente la Line 5 de l'Algorithm 1 de LLM-TIKG (Hu et al., 2024) :
    traitement regex séparé du bloc IoC, sans passer par le LLM.
    """
    relations = []
    main_object = os.path.splitext(os.path.basename(source_file))[0]

    # Hashes MD5 (32) / SHA1 (40) / SHA256 (64)
    for h in re.findall(r"\b[0-9a-fA-F]{32,64}\b", ioc_block):
        relations.append((main_object, "has_ioc_hash", h))

    # IPs (normales et obfusquées : 1[.]2[.]3[.]4)
    for ip in re.findall(r"\b(?:\d{1,3}[\[\.]){3}\d{1,3}\b", ioc_block):
        clean_ip = ip.replace("[.]", ".").replace("[", "").replace("]", "")
        relations.append((main_object, "has_ioc_ip", clean_ip))

    # Domaines obfusqués : evil[.]com
    for domain in re.findall(r"[\w.\-]+\[\.\][\w]{2,6}", ioc_block):
        clean_domain = domain.replace("[.]", ".")
        relations.append((main_object, "has_ioc_domain", clean_domain))

    # CVE identifiers
    for cve in re.findall(r"CVE-\d{4}-\d{4,7}", ioc_block, re.IGNORECASE):
        relations.append((main_object, "has_ioc_cve", cve.upper()))

    return relations


def save_results(doc: ProcessedDocument, output_dir: str) -> None:
    """Sauvegarde les chunks et IoC relations en JSON (un fichier par document)."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(doc.source_file))[0]
    out_path = os.path.join(output_dir, f"{base}_processed.json")

    payload = {
        "source_file": doc.source_file,
        "n_paragraphs": len(doc.paragraphs),
        "n_chunks": len(doc.chunks),
        "chunks": doc.chunks,
        "ioc_relations": [list(t) for t in doc.ioc_relations],
        "error": doc.error,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"  → Sauvegardé : {out_path}")


# ── Traitement d'un fichier ───────────────────────────────────────────────────

def process_single_file(
    file_path: str,
    cleaner: CTIDataPipeline,
    chunking_cfg: ChunkingConfig,
    output_dir: Optional[str] = None,
) -> ProcessedDocument:
    """
    Orchestre le pipeline complet pour un fichier :

        Étape 1 — CTIDataPipeline.process_file()
                  extraction → normalisation → séparation IoC/prose
                  → sanitisation IoCs → anonymisation PII → boilerplate

        Étape 2 — split_into_paragraphs()
                  Stage 1 chunking : découpage structurel (\n\n)

        Étape 3 — semantic_chunking_improved()
                  Stage 2 chunking : raffinement sémantique

        Étape 4 — process_ioc_block()
                  Extraction regex du bloc IoC (triplets Neo4j-ready)

        Étape 5 — save_results() [optionnel]
                  Sérialisation JSON si --output fourni
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

    # ── Étape 4 : IoC block (regex, hors LLM) ────────────────────────────────
    ioc_relations = process_ioc_block(ioc_block, file_path) if ioc_block else []
    if ioc_relations:
        log.info(f"  {len(ioc_relations)} IoC relations extraites")

    # ── Construction du résultat ──────────────────────────────────────────────
    doc = ProcessedDocument(
        source_file=file_path,
        prose_clean=prose_clean,
        ioc_block=ioc_block,
        paragraphs=paragraphs,
        chunks=chunks,
        ioc_relations=ioc_relations,
    )

    # ── Étape 5 : Sauvegarde (optionnelle) ───────────────────────────────────
    if output_dir:
        save_results(doc, output_dir)

    log.info(f"  ✓ {doc.summary()}")
    return doc


# ── Traitement en lot ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def process_directory(
    input_dir: str,
    cleaner: CTIDataPipeline,
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

    # Résumé global
    ok = [r for r in results if not r.error]
    ko = [r for r in results if r.error]
    total_chunks = sum(len(r.chunks) for r in ok)
    log.info(
        f"\n{'─' * 50}\n"
        f"Résumé : {len(ok)}/{len(results)} fichiers OK | "
        f"{total_chunks} chunks au total | "
        f"{len(ko)} erreurs"
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
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
        "--theta_s", type=float, default=0.5,
        help="Seuil similarité sémantique (défaut : 0.5)"
    )
    parser.add_argument(
        "--theta_e", type=float, default=0.1,
        help="Seuil overlap entités Jaccard (défaut : 0.1)"
    )
    parser.add_argument(
        "--l_max", type=int, default=400,
        help="Longueur max d'un chunk en mots (défaut : 400)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Instanciation unique (Presidio + spaCy chargés une seule fois)
    log.info("Initialisation du pipeline de nettoyage (Presidio + spaCy)...")
    cleaner = CTIDataPipeline()

    cfg = ChunkingConfig(
        theta_s=args.theta_s,
        theta_e=args.theta_e,
        l_max=args.l_max,
    )

    input_path = args.input

    if os.path.isfile(input_path):
        # ── Mode fichier unique ────────────────────────────────────────────────
        result = process_single_file(input_path, cleaner, cfg, args.output)

        print(f"\n{'═' * 60}")
        print(f"Fichier : {result.source_file}")
        print(f"{'═' * 60}")
        for i, chunk in enumerate(result.chunks, 1):
            preview = chunk[:120].replace("\n", " ")
            print(f"\n[Chunk {i:02d}] {preview}...")
        if result.ioc_relations:
            print(f"\n{'─' * 60}")
            print(f"IoC relations ({len(result.ioc_relations)}) :")
            for rel in result.ioc_relations[:10]:
                print(f"  {rel[0]} --[{rel[1]}]--> {rel[2]}")

    elif os.path.isdir(input_path):
        # ── Mode dossier ──────────────────────────────────────────────────────
        process_directory(input_path, cleaner, cfg, args.output)

    else:
        log.error(f"Chemin invalide : {input_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
