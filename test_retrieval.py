from dotenv import load_dotenv
import os
from app.retrieval import retrieve_context

# Load .env for DATABASE_URL and CHROMA_DB_PATH
load_dotenv()

def simple_test():
    hits = retrieve_context("hypertension management", top_k=2)
    print("Retrieved chunks:", hits)

if __name__ == "__main__":
    simple_test()
