import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use Gemini Flash (free & fast)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def get_detected_role(text):
    prompt = f"""
    You are a medical recruitment assistant.
    Based on this CV, what is the person's current medical role or specialty?
    Please return just the role (e.g., Pediatrician, ENT Surgeon, Cardiologist).

    CV:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text.strip()
