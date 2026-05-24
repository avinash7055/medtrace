import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from knowledge_base.loader import load_json_qa_files, qa_pairs_to_documents, chunk_documents

print("Loading documents...")
pairs = load_json_qa_files()
docs = qa_pairs_to_documents(pairs)
chunks = chunk_documents(docs)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
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
