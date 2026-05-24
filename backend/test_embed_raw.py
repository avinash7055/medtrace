import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

api_key = os.getenv("GEMINI_API_KEY")
texts = ["Hello world", "How are you?", "Testing embeddings"]

print("--- Testing gemini-embedding-001 ---")
try:
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    res_lc = embeddings_model.embed_documents(texts)
    print(f"gemini-embedding-001 returned length: {len(res_lc)}")
except Exception as e:
    print(f"gemini-embedding-001 failed: {e}")

print("\n--- Testing gemini-embedding-2 ---")
try:
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key,
    )
    res_lc = embeddings_model.embed_documents(texts)
    print(f"gemini-embedding-2 returned length: {len(res_lc)}")
except Exception as e:
    print(f"gemini-embedding-2 failed: {e}")
