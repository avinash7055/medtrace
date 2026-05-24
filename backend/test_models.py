import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

try:
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API Key present: {bool(api_key)}")
    
    client = genai.Client(api_key=api_key)
    print("Listing all models from Gemini API...")
    
    models = list(client.models.list())
    embedding_models = [m.name for m in models if "embed" in m.name.lower() or any("embed" in act.lower() for act in m.supported_actions)]
    
    print("\n--- Available Embedding Models ---")
    for model_name in embedding_models:
        print(f" - {model_name}")
        
    print("\n--- All Available Models ---")
    for m in models[:15]:
        print(f" - {m.name} | Supported Actions: {m.supported_actions}")
        
except Exception as e:
    print(f"Error during API call: {e}")
