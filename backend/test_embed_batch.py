import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

texts = [f"This is sample document chunk number {i}" for i in range(50)]

print("Sending 50 documents in a single API call via raw SDK...")
try:
    res = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts
    )
    print(f"Raw SDK response status: Success!")
    print(f"Number of embeddings returned: {len(res.embeddings)}")
except Exception as e:
    print(f"Failed: {e}")
