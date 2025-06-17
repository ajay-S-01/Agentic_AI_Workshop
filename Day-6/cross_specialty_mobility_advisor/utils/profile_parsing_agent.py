import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def extract_raw_text_from_pdf(path):
    """Extract raw text from PDF pages using PyMuPDF"""
    try:
        doc = fitz.open(path)
        text = " ".join([page.get_text() for page in doc])
        return text.strip()
    except Exception as e:
        print("❌ Failed to extract PDF text:", e)
        return ""

def extract_text_from_pdf(path):
    """Gemini-powered PDF parser to extract relevant medical CV info"""
    raw_text = extract_raw_text_from_pdf(path)

    if not raw_text:
        return "Unable to extract text from the uploaded PDF."

    prompt = f"""
    You are a helpful AI that extracts and cleans relevant information from a doctor's CV.

    Based on the following raw text from a PDF CV, summarize clearly:
    - Professional Summary
    - Education
    - Certifications
    - Experience
    - Skills

    If some sections are missing, skip them.

    Return the result in a clean text format with section headings.

    RAW TEXT:
    {raw_text}
    """

    try:
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip()
        return cleaned_text
    except Exception as e:
        print("❌ Gemini parsing failed:", e)
        return raw_text  # fallback to raw if Gemini fails
