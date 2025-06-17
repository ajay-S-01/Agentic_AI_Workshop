import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def recommend_micro_certifications(current_role: str, target_role: str) -> list:
    prompt = f"""
    You are a career advisor for medical professionals.

    A doctor with the role '{current_role}' wants to become a '{target_role}'.
    Recommend 3 to 5 short certifications, CME courses, or upskilling programs they should take to bridge the skill gap.

    Format the output as a numbered list of certification names with short descriptions (not full paragraphs).
    """
    response = model.generate_content(prompt)
    return response.text.strip()
