import os
import urllib.request
import json
from dotenv import load_dotenv
load_dotenv()

try:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"Connecting to Ollama at: {ollama_url}")
    
    req = urllib.request.Request(f"{ollama_url}/api/tags")
    with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
        models = data.get("models", [])
        
    print(f"\nFound {len(models)} local Ollama models:")
    for m in models:
        name = m.get("name", "unknown")
        details = m.get("details", {})
        family = details.get("family", "unknown")
        parameter_size = details.get("parameter_size", "unknown")
        print(f" - {name} (family: {family}, size: {parameter_size})")
        
except Exception as e:
    print(f"Error checking Ollama models: {e}")

