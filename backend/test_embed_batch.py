import os
from dotenv import load_dotenv
load_dotenv()

ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
texts = [f"This is sample document chunk number {i}" for i in range(50)]

print("Sending 50 documents to local Ollama embedding API...")
try:
    from langchain_ollama import OllamaEmbeddings
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=ollama_url,
    )
    res = embeddings_model.embed_documents(texts)
    print(f"Ollama response status: Success!")
    print(f"Number of embeddings returned: {len(res)}")
except Exception as e:
    print(f"Failed: {e}")

