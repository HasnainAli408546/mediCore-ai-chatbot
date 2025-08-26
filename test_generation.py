from dotenv import load_dotenv
import os
from app.generation import generate_medical_answer, generate_fallback_response

load_dotenv()

# Dummy context for testing
context_chunks = [
    {
        "id": "guideline1-0",
        "text": "Patients with hypertension should undergo lifestyle modifications including diet and exercise.",
        "source": "guideline1.pdf",
        "relevance_score": 0.95
    }
]

question = "How do I manage hypertension?"

print("=== With Context ===")
print(generate_medical_answer(question, context_chunks))

print("\n=== No Context ===")
print(generate_fallback_response(question))
