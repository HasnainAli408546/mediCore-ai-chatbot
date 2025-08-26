import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Test the connection
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say hello and confirm you're working!")
    print("✅ Gemini API Test Successful!")
    print(f"Response: {response.text}")
except Exception as e:
    print("❌ Gemini API Test Failed!")
    print(f"Error: {e}")
