import sqlite3
import os

def split_large_tram_report(db_path: str, output_dir: str, sentences_per_doc: int = 150):
    """
    Extrait les 11 000+ phrases de la base TRAM et les découpe en plusieurs
    fichiers .txt de taille raisonnable pour alimenter l'InputData.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # On récupère toutes les phrases dans l'ordre chronologique/logique
        cursor.execute("SELECT text FROM tram_sentence ORDER BY id")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("[!] Aucune phrase trouvée dans tram_sentence.")
            return 0

        # Regroupement des phrases par blocs (ex: 150 phrases par document)
        doc_counter = 1
        current_sentences = []

        for row in rows:
            text = row[0]
            if text and text.strip():
                current_sentences.append(text.strip())

            # Dès qu'on atteint le quota par document, on écrit un fichier .txt
            if len(current_sentences) >= sentences_per_doc:
                file_path = os.path.join(output_dir, f"tram_corpus_doc_{doc_counter}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(current_sentences))

                doc_counter += 1
                current_sentences = []

        # Écriture du dernier bloc s'il reste des phrases
        if current_sentences:
            file_path = os.path.join(output_dir, f"tram_corpus_doc_{doc_counter}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(" ".join(current_sentences))
            doc_counter += 1

        print(f"[SUCCÈS] Corpus généré : {doc_counter - 1} documents textuels créés dans :\n-> {output_dir}")
        return doc_counter - 1

    except Exception as e:
        print(f"[ERREUR CRITIQUE] {e}")
        return 0

if __name__ == "__main__":
    CHEMIN_BASE_TRAM = r"C:\Users\cleme\IdeaProjects\tram\data\db.sqlite3"
    DOSSIER_INPUT = r"C:\Users\cleme\IdeaProjects\TemporalEntityRelationExtractionPipeline\Main\InputData"

    split_large_tram_report(CHEMIN_BASE_TRAM, DOSSIER_INPUT, sentences_per_doc=150)