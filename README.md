# Temporal Entity Relation Extraction Pipeline

Pipeline de nettoyage et de chunking sémantique de rapports CTI (Cyber Threat Intelligence).

## Prérequis

- **Python 3.10+** (testé avec Python 3.12)

## Installation

### 1. Créer l'environnement virtuel

```powershell
py -3.12 -m venv .venv
```

### 2. Activer l'environnement

```powershell
# Windows
.venv\Scripts\activate
```

### 3. Installer les dépendances

```powershell
pip install -r requirements.txt
```

### 4. Télécharger le modèle spaCy

```powershell
python -m spacy download en_core_web_sm
```

> **Note :** Au premier lancement, le modèle d'embedding `BAAI/bge-m3` (~400 MB) sera téléchargé automatiquement depuis HuggingFace.

## Utilisation

Placer les fichiers `.pdf` ou `.txt` à traiter dans le dossier `InputData/`, puis lancer :

```powershell
.venv\Scripts\python dataCleaning\CTIOrchestrator.py -i InputData -o OutputChunks
```

Les résultats JSON sont générés dans `OutputChunks/`.

### Options disponibles

| Option | Défaut | Description |
|---|---|---|
| `-i` / `--input` | — | Fichier unique ou dossier source (**requis**) |
| `-o` / `--output` | — | Dossier de sortie JSON (optionnel) |
| `--theta_s` | `0.2` | Seuil de similarité sémantique cosinus |
| `--theta_e` | `0.15` | Seuil de chevauchement d'entités (Jaccard) |
| `--l_max` | `400` | Longueur maximale d'un chunk (en mots) |

## Structure du projet

```
├── dataCleaning/
│   ├── CTIOrchestrator.py       # Point d'entrée principal
│   ├── CTITextCleaner.py        # Extraction, nettoyage, anonymisation
│   ├── SemanticChunk.py         # Chunking sémantique (SC-LKM)
│   └── MitreWhitelistLoader.py  # Whitelist dynamique MITRE ATT&CK
├── InputData/                   # Rapports CTI sources (non versionné)
├── OutputChunks/                # Résultats JSON (non versionné)
└── requirements.txt
```
