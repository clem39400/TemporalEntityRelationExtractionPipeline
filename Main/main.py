import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# --- IMPORTS DE TES BRIQUES ---
from Main.CTIDocumentExtractor import CTIDocumentExtractor
from Main.CTISemanticChunker import CTISemanticChunker

# Nouveaux imports pour la Phase 3
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
    "output_directory": "C:\\Users\\cleme\\IdeaProjects\\TemporalEntityRelationExtractionPipeline\\Main\\ExtractedResults",

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
    model = genai.GenerativeModel(config["model_name"])

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
            # Assure-toi d'avoir importé parse_json_from_response !
            parsed_data = parse_json_from_response(raw_response)
        except Exception as e:
            print(f"Erreur API/Parsing sur le chunk global {global_chunk_id} : {e}")
            parsed_data = None

        results.append({
            "chunk_metadata": {
                "source": chunk['source'],
                "chunk_id": global_chunk_id  # On sauvegarde l'ID entier pour la Phase 3
            },
            "hyperparameters": config,
            "extraction": parsed_data
        })

        time.sleep(15) # Pause Anti-Rate Limit

    # Sauvegarde des extractions locales
    output_file = os.path.join(config["output_directory"], f"results_{config['prompt_type']}_{config['model_name'].replace('.', '-')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[SUCCÈS] Phase 2 terminée. Résultats bruts dans : {output_file}")

    # --- PHASE 3 : RÉCONCILIATION, NETTOYAGE ET VISUALISATION ---
    if config["run_phase_3"]:
        print(f"\n{'='*50}")
        print("PHASE 3 : CONSTRUCTION DU GRAPHE DE CONNAISSANCES")
        print(f"{'='*50}\n")

        # Définition des chemins pour cette exécution spécifique
        reconciled_path = os.path.join(config["output_directory"], f"Reconciled_{config['prompt_type']}.json")
        cleaned_path = os.path.join(config["output_directory"], f"Cleaned_{config['prompt_type']}.json")
        viz_path = os.path.join(config["output_directory"], f"Viz_{config['prompt_type']}.html")

        # 3.1 Réconciliation
        print("[*] 3.1 Lancement de la Réconciliation...")
        reconcile_graph(
            input_path=output_file,
            output_path=reconciled_path,
            similarity_threshold=config["reconciliation_sim_threshold"],
            malware_threshold=config["reconciliation_malware_threshold"]
        )

        # 3.2 Nettoyage
        print("[*] 3.2 Lancement du Nettoyage dynamique...")
        dynamic_cleaner(
            input_path=reconciled_path,
            output_path=cleaned_path,
            max_connectivity=config["cleaner_max_connectivity"]
        )

        # 3.3 Visualisation
        print("[*] 3.3 Génération de la visualisation HTML...")
        visualize_graph(
            json_path=cleaned_path,
            output_html=viz_path
        )

        print(f"\n[SUCCÈS TOTAL] Graphe final généré ici : {viz_path}")


if __name__ == "__main__":
    run_full_pipeline(CONFIG)