import sqlite3

def extract(DB_PATH) :
    conn = sqlite3.connect(DB_PATH)
    print("Nombre de phrases :", conn.execute("SELECT COUNT(*) FROM tram_sentence").fetchone()[0])
    print("Nombre d'annotations (mappings) :", conn.execute("SELECT COUNT(*) FROM tram_mapping").fetchone()[0])
    print("Nombre de rapports :", conn.execute("SELECT COUNT(*) FROM tram_report").fetchone()[0])
    conn.close()


if __name__ == "__main__":
    DB_PATH = r"C:\Users\cleme\IdeaProjects\TemporalEntityRelationExtractionPipeline\Main\Data\db.sqlite3"
    extract(DB_PATH)