import os
import google.generativeai as genai
from dotenv import load_dotenv

# Charge la clé depuis votre fichier .env
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print("Modèles capables de générer du texte :")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"-> {m.name}")