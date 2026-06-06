import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from knowledge_base.loader import load_json_qa_files, qa_pairs_to_documents, chunk_documents

print("Loading documents...")
pairs = load_json_qa_files()
docs = qa_pairs_to_documents(pairs)
chunks = chunk_documents(docs)

ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=ollama_url,
)

test_batch = [c.page_content for c in chunks[:10]]
print(f"Embedding a test batch of {len(test_batch)} items...")
try:
    res = embeddings_model.embed_documents(test_batch)
    print(f"Success! Embedded count: {len(res)}")
    if len(res) > 0:
        print(f"Dimensions of first embedding: {len(res[0])}")
except Exception as e:
    print(f"Error during langchain embedding: {e}")

