import os
import urllib.request
import json
from dotenv import load_dotenv
load_dotenv()

ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
texts = ["Hello world", "How are you?", "Testing embeddings"]

print("--- Testing raw Ollama /api/embeddings ---")
try:
    # Ollama legacy embeddings endpoint (single string)
    req = urllib.request.Request(
        f"{ollama_url}/api/embeddings",
        data=json.dumps({
            "model": "nomic-embed-text",
            "prompt": texts[0]
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        res = json.loads(response.read().decode("utf-8"))
        embedding = res.get("embedding", [])
        print(f"Success! /api/embeddings returned length: {len(embedding)}")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Testing raw Ollama /api/embed ---")
try:
    # Ollama modern embed endpoint (batch input)
    req = urllib.request.Request(
        f"{ollama_url}/api/embed",
        data=json.dumps({
            "model": "nomic-embed-text",
            "input": texts
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        res = json.loads(response.read().decode("utf-8"))
        embeddings = res.get("embeddings", [])
        print(f"Success! /api/embed returned {len(embeddings)} embeddings of length {len(embeddings[0]) if embeddings else 0}")
except Exception as e:
    print(f"Failed: {e}")

