import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def generate_specialty_graph(current_role: str) -> list:
    prompt = f"""
    You are a medical career advisor AI.

    A doctor is currently working as a '{current_role}'.
    Suggest 3-5 lateral or adjacent medical specialties they can transition into.

    For each suggested role, include:
    - suggested role name
    - a skill overlap score (0.0 to 1.0)
    - 1-2 bridging certifications they would need

    Format the result as a JSON list with this structure:
    [
      {{
        "from": "{current_role}",
        "to": "SuggestedRole1",
        "weight": 0.85,
        "certifications": ["Cert A", "Cert B"]
      }},
      ...
    ]
    Only return the JSON. Do not include explanations or extra text.
    """

    try:
        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        # For debugging
        print("\n🤖 Gemini Output:\n", raw_output)

        # Clean the response - remove markdown code blocks if present
        cleaned_output = raw_output
        if "```json" in raw_output:
            # Extract JSON from markdown code block
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
            if json_match:
                cleaned_output = json_match.group(1)
        elif "```" in raw_output:
            # Handle generic code blocks
            json_match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
            if json_match:
                cleaned_output = json_match.group(1)

        # Try parsing the cleaned JSON
        data = json.loads(cleaned_output)
        
        # Validate the structure
        if isinstance(data, list) and len(data) > 0:
            # Ensure each item has required fields
            for item in data:
                if not all(key in item for key in ["from", "to", "weight", "certifications"]):
                    print("⚠️ Warning: Missing required fields in suggestion")
                    continue
            return data
        else:
            print("❌ Invalid data structure returned")
            return []
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw output: {raw_output}")
        return []
    except Exception as e:
        print(f"❌ API or other error: {e}")
        return []

# Fallback function to provide sample data if API fails
def get_fallback_recommendations(current_role: str) -> list:
    """Provide fallback recommendations when API fails"""
    fallback_data = {
        "General Practitioner": [
            {
                "from": current_role,
                "to": "Family Medicine Specialist",
                "weight": 0.9,
                "certifications": ["Family Medicine Board Certification", "Preventive Care CME"]
            },
            {
                "from": current_role,
                "to": "Internal Medicine",
                "weight": 0.8,
                "certifications": ["Internal Medicine Residency", "Adult Care Specialization"]
            }
        ],
        "Cardiologist": [
            {
                "from": current_role,
                "to": "Cardiac Surgeon",
                "weight": 0.7,
                "certifications": ["Surgical Training", "Cardiothoracic Surgery Fellowship"]
            },
            {
                "from": current_role,
                "to": "Interventional Cardiologist",
                "weight": 0.85,
                "certifications": ["Interventional Cardiology Fellowship", "Catheterization Lab Training"]
            }
        ]
    }
    
    return fallback_data.get(current_role, [
        {
            "from": current_role,
            "to": "Related Specialty",
            "weight": 0.75,
            "certifications": ["Relevant Certification", "Continuing Medical Education"]
        }
    ])