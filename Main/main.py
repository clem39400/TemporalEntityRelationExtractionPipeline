import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# --- IMPORTS DE TES BRIQUES ---
from Main.CTIDocumentExtractor import CTIDocumentExtractor
from Main.CTISemanticChunker import CTISemanticChunker
from Main.GraphReconcilier import reconcile_graph
from Main.GraphCleaner import dynamic_cleaner
from Main.VizualizeGraph import visualize_graph
from Main.Utils import parse_json_from_response, get_rocade_few_shot_prompt, get_cot_prompt

# ==========================================
# 1. HYPERPARAMÈTRES (CONFIGURATION CENTRALE)
# ==========================================

CONFIG = {
    # --- Chemins ---
    "input_directory": "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main\\InputData",
    "output_directory": "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main\\ExtractedResults2",

    # --- Phase 1 & 2 : Extraction et LLM ---
    "model_name": "gemini-3.1-flash-lite",
    "prompt_type": "cot",             # "few_shot" ou "cot"
    "use_chunking": True,
    "use_rocade": True,
    "test_limit": 300,                 # 0 pour tout traiter

    # --- Phase 3 : Construction du Graphe ---
    "run_phase_3": True,              # Activer/Désactiver toute la phase 3
    "reconciliation_sim_threshold": 0.85,
    "reconciliation_malware_threshold": 0.95,
    "cleaner_max_connectivity": 15
}

# ==========================================
# 2. ORCHESTRATEUR GLOBAL
# ==========================================

def run_full_pipeline(config: dict):
    print(f"\n{'='*50}")
    print(f"DÉMARRAGE DE LA PIPELINE (Modèle: {config['model_name']} | Prompt: {config['prompt_type']})")
    print(f"{'='*50}\n")

    os.makedirs(config["output_directory"], exist_ok=True)

    # --- INITIALISATION API ---
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("La variable GEMINI_API_KEY est introuvable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        config["model_name"],
        generation_config=genai.GenerationConfig(
            temperature=0.0,  # Bloque toute créativité/variabilité aléatoire
            top_k=1,          # Ne sélectionne systématiquement que le token le plus probable
            top_p=0.0      # Restreint drastiquement l'espace d'échantillonnage
        )
    )

    # --- INITIALISATION DES OUTILS ---
    extractor = CTIDocumentExtractor(remove_non_ascii=True, hide_sensitive=True, margin_tolerance=0.08)
    if config["use_chunking"]:
        chunker = CTISemanticChunker(model_name='all-MiniLM-L6-v2', similarity_threshold=0.35, device='cpu')

    all_chunks = []

    # --- PHASE 1 : LECTURE ET PRÉPARATION DES DONNÉES ---
    print(f"[*] Lecture des fichiers dans : {config['input_directory']}")
    for filename in os.listdir(config["input_directory"]):
        file_path = os.path.join(config["input_directory"], filename)
        if not os.path.isfile(file_path): continue

        clean_text = extractor.extract_file(file_path)
        if not clean_text: continue

        if config["use_chunking"]:
            file_chunks = chunker.chunk_report(clean_text, source_filename=filename)
            all_chunks.extend(file_chunks)
        else:
            all_chunks.append({
                "source": filename,
                "chunk_id": 0,
                "text": clean_text,
                "sentence_count": len(clean_text.split('.'))
            })

    print(f"[*] Total de fragments (chunks) préparés : {len(all_chunks)}")
    chunks_to_process = all_chunks[:config["test_limit"]] if config["test_limit"] > 0 else all_chunks

    # --- PHASE 2 : EXTRACTION LLM ---
    results = []
    for global_chunk_id, chunk in enumerate(chunks_to_process):

        print(f"\nAnalyse {global_chunk_id+1}/{len(chunks_to_process)} (Source: {chunk['source']} | ID: {chunk['chunk_id']})...")

        prompt = ""
        if config["use_rocade"]:
            if config["prompt_type"].lower() == "cot":
                prompt = get_cot_prompt(chunk['text'], global_chunk_id)
            elif config["prompt_type"].lower() == "few_shot":
                prompt = get_rocade_few_shot_prompt(chunk['text'], global_chunk_id)

        try:
            raw_response = model.generate_content(prompt).text
            parsed_data = parse_json_from_response(raw_response)
        except Exception as e:
            print(f"Erreur API/Parsing sur le chunk global {global_chunk_id} : {e}")
            parsed_data = None

        results.append({
            "chunk_metadata": {
                "source": chunk['source'],
                "chunk_id": global_chunk_id
            },
            "hyperparameters": config,
            "extraction": parsed_data
        })

        time.sleep(15) # Pause Anti-Rate Limit

    # Sauvegarde des extractions locales globales (Backup)
    output_file = os.path.join(config["output_directory"], f"results_{config['prompt_type']}_{config['model_name'].replace('.', '-')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[SUCCÈS] Phase 2 terminée. Résultats bruts globaux dans : {output_file}")

    # --- PHASE 3 : CONSTRUCTION DES GRAPHES PAR DOCUMENT ---
    if config["run_phase_3"]:
        print(f"\n{'='*50}")
        print("PHASE 3 : RÉCONCILIATION ET CONSTRUCTION PAR DOCUMENT")
        print(f"{'='*50}\n")

        # Grouper les résultats bruts par fichier source
        results_by_source = {}
        for res in results:
            src = res["chunk_metadata"]["source"]
            if src not in results_by_source:
                results_by_source[src] = []
            results_by_source[src].append(res)

        # Traiter chaque document indépendamment
        for source_file, source_data in results_by_source.items():
            # Créer un nom de fichier propre sans extension
            safe_source_name = os.path.splitext(source_file)[0]
            print(f"[*] Génération du graphe pour : {safe_source_name}")

            raw_source_path = os.path.join(config["output_directory"], f"raw_{safe_source_name}.json")
            reconciled_path = os.path.join(config["output_directory"], f"Reconciled_{safe_source_name}.json")
            cleaned_path = os.path.join(config["output_directory"], f"Cleaned_{safe_source_name}.json")
            viz_path = os.path.join(config["output_directory"], f"Viz_{safe_source_name}.html")

            # Sauvegarder un JSON brut temporaire pour ce document précis
            with open(raw_source_path, 'w', encoding='utf-8') as f:
                json.dump(source_data, f, indent=4, ensure_ascii=False)

            # 3.1 Réconciliation
            reconcile_graph(
                input_path=raw_source_path,
                output_path=reconciled_path,
                similarity_threshold=config["reconciliation_sim_threshold"],
                malware_threshold=config["reconciliation_malware_threshold"]
            )

            # 3.2 Nettoyage
            dynamic_cleaner(
                input_path=reconciled_path,
                output_path=cleaned_path,
                max_connectivity=config["cleaner_max_connectivity"]
            )

            # 3.3 Visualisation
            visualize_graph(
                json_path=cleaned_path,
                output_html=viz_path
            )

            print(f"    -> Terminé : {reconciled_path}")

        print(f"\n[SUCCÈS TOTAL] Tous les graphes documentaires ont été générés !")

if __name__ == "__main__":
    run_full_pipeline(CONFIG)