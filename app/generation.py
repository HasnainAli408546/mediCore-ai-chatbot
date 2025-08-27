import os
import logging
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables and configure logging
load_dotenv()
logger = logging.getLogger(__name__)

# Configure Google Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
logger.info("Google Generative AI configured successfully")

def generate_medical_answer(query: str, context_chunks: List[Dict]) -> str:
    """
    Generate a medical answer using retrieved context and Google Gemini
    """
    logger.info(f"Generating medical answer for query: '{query[:100]}...' using {len(context_chunks)} context chunks")
    
    if not context_chunks:
        logger.warning("No context chunks provided, falling back to general response")
        return generate_fallback_response(query)
    
    # Build context from chunks
    context_text = ""
    for chunk in context_chunks:
        context_text += f"Source: {chunk['source']}\n"
        context_text += f"Content: {chunk['text']}\n\n"
    
    logger.info(f"Built context from {len(context_chunks)} chunks, total context length: {len(context_text)} chars")
    
    # Create prompt
    prompt = f"""
You are a helpful medical AI assistant. Based on the provided medical context, answer the user's question accurately and safely.

IMPORTANT GUIDELINES:
1. Only use information from the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Always recommend consulting healthcare professionals for medical decisions
4. Never provide specific dosing or treatment recommendations without professional oversight
5. Be clear about limitations and uncertainties

CONTEXT:
{context_text}

USER QUESTION: {query}

ANSWER:"""

    try:
        logger.info("Sending request to Google Gemini API...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if response.text:
            logger.info(f"Successfully generated response of {len(response.text)} characters")
            return response.text
        else:
            logger.error("Empty response received from Gemini API")
            return generate_fallback_response(query)
            
    except Exception as e:
        logger.error(f"Error generating medical answer with Gemini: {e}")
        return generate_fallback_response(query)

def generate_fallback_response(query: str) -> str:
    """
    Generate a safe fallback response when context-based generation fails
    """
    logger.warning(f"Generating fallback response for query: '{query[:50]}...'")
    
    fallback_prompt = f"""
You are a medical AI assistant. The user asked: "{query}"

Since specific medical context is not available, provide a general, safe response that:
1. Acknowledges their question
2. Provides general health information if appropriate
3. Strongly recommends consulting healthcare professionals
4. Mentions limitations without specific medical context

Keep the response helpful but conservative and safe.

ANSWER:"""

    try:
        logger.info("Sending fallback request to Google Gemini API...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(fallback_prompt)
        
        if response.text:
            logger.info(f"Successfully generated fallback response of {len(response.text)} characters")
            return response.text
        else:
            logger.error("Failed to generate fallback response from Gemini API")
            return "I apologize, but I'm unable to provide a response at this time. Please consult with a healthcare professional for medical advice."
            
    except Exception as e:
        logger.error(f"Error generating fallback response: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please consult with a healthcare professional for medical advice."
