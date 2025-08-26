import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
from .retrieval import format_context_for_prompt
import logging

load_dotenv()
logger = logging.getLogger(__name__)
# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

def generate_medical_answer(
    question: str,
    context_chunks: List[Dict],
    max_tokens: int = 500
) -> str:
    """
    Use Google Gemini to generate a medical answer from context chunks.
    """
    # Format retrieved context
    context_text = format_context_for_prompt(context_chunks)

    # System prompt with guardrails
    system_prompt = (
        "You are mediCore AI, a medical assistant chatbot. Follow these rules:\n"
        "1. Use ONLY the provided clinical sources.\n"
        "2. Cite sources using [Source X] notation.\n"
        "3. If data is insufficient, state limitations clearly.\n"
        "4. Advise consulting healthcare professionals for personalized care.\n"
        "5. Maintain professional, evidence-based tone.\n"
        "6. Acknowledge uncertainty when sources conflict.\n"
    )

    # User prompt combining context and question
    user_prompt = (
        f"Clinical Context:\n{context_text}\n\n"
        f"Patient Question: {question}\n\n"
        "Provide an evidence-based response using ONLY the provided sources."
    )

    try:
        # Call Gemini to generate content
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.2,
                top_p=0.8,
                top_k=20
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini generation error: %s", str(e))
        return (
            "I’m sorry, I encountered an error generating a response. "
            "Please consult a healthcare professional for medical advice."
        )

def generate_fallback_response(question: str) -> str:
    """
    Provide a fallback when no relevant context is found.
    """
    return (
        f"I don't have sufficient clinical context to answer your question about \"{question}\". "
        "For accurate medical guidance, please consult a qualified healthcare professional."
    )
