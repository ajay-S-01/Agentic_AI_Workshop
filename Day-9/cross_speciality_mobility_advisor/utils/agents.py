# utils/agents.py
import os
import json
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

from utils.rag_utils import get_retriever, setup_vectordb

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- Tool 1: Profile Parsing ---
@tool
def extract_cv_info(pdf_path: str) -> str:
    """Extracts and cleans relevant information (Professional Summary, Education, Certifications, Experience, Skills) from a doctor's CV in PDF format."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        raw_text = " ".join([doc.page_content for doc in documents])
    except Exception as e:
        return f"Error loading PDF: {e}"

    if not raw_text:
        return "Unable to extract text from the uploaded PDF."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI that extracts and cleans relevant information from a doctor's CV."),
        ("user", """Based on the following raw text from a PDF CV, summarize clearly:
        - Professional Summary
        - Education
        - Certifications
        - Experience
        - Skills

        If some sections are missing, skip them.
        Return the result in a clean text format with section headings.

        RAW TEXT:
        {raw_text}
        """)
    ])
    chain = prompt | llm
    response = chain.invoke({"raw_text": raw_text})
    return response.content.strip()

# --- Tool 2: Role Detection ---
@tool
def detect_medical_role(cv_text: str) -> str:
    """Detects the current medical role or specialty from a given CV text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical recruitment assistant."),
        ("user", """Based on this CV, what is the person's current medical role or specialty?
        Please return just the role (e.g., Pediatrician, ENT Surgeon, Cardiologist).

        CV:
        {cv_text}
        """)
    ])
    chain = prompt | llm
    response = chain.invoke({"cv_text": cv_text})
    return response.content.strip()

# --- Tool 3: Specialty Proximity Mapper (RAG-Enabled) ---
# This tool will utilize RAG to get context
@tool
def get_lateral_specialty_recommendations(current_role: str, cv_text: str) -> str:
    """Suggests 3-5 lateral or adjacent medical specialties for a doctor based on their current role and provided CV information, using RAG for context."""

    retriever = get_retriever()
    if not retriever:
        print("DEBUG: Retriever not initialized, returning error.") # New debug print
        return json.dumps({"error": "RAG system not initialized. Please ensure data is available for RAG."})

    # First, get relevant context from RAG
    rag_query = f"Medical taxonomy and career transition case studies for a {current_role} looking for lateral moves based on their CV: {cv_text[:500]}..." # Limit CV text for query

    print(f"DEBUG: Performing retrieval for query: {rag_query[:100]}...") # New debug print
    docs = retriever.invoke(rag_query)
    print(f"DEBUG: Retrieved {len(docs)} documents.") # New debug print

    # Create a prompt that incorporates the retrieved documents
    prompt_template = ChatPromptTemplate.from_template(
        """You are a medical career advisor AI. Use the following retrieved medical taxonomy and career transition case studies to inform your suggestions:
        {context}

        A doctor is currently working as a '{current_role}'. Based on their CV information and the provided context,
        suggest 3-5 lateral or adjacent medical specialties they can transition into.

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
    )

    print("DEBUG: Creating document chain.") # New debug print
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    print("DEBUG: Creating retrieval chain.") # New debug print
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        print("DEBUG: Invoking retrieval chain.") # New debug print
        response = retrieval_chain.invoke({"current_role": current_role, "cv_text": cv_text})
        print("DEBUG: Retrieval chain invoked. Raw response:", response) # New debug print
        # The response from create_retrieval_chain will have 'answer' and 'context'
        raw_output = response["answer"].strip()

        # For debugging (make sure this is not suppressed)
        print("\n🤖 Gemini Output Raw Before Cleaning:\n", raw_output)

        # Clean the response - remove markdown code blocks if present
        # ... (rest of the cleaning logic)
        print("DEBUG: JSON parsing initiated.") # New debug print
        data = json.loads(cleaned_output)
        print("DEBUG: JSON parsed successfully.") # New debug print
        # ... (rest of the function)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in get_lateral_specialty_recommendations: {e}")
        print(f"Raw output causing error: {raw_output}")
        return json.dumps({"error": f"JSON parsing failed: {e}"})
    except Exception as e:
        print(f"❌ API or other error in get_lateral_specialty_recommendations: {e}")
        # print("DEBUG: Full traceback:", traceback.format_exc()) # If you import traceback
        return json.dumps({"error": f"Internal error: {e}"})

# --- Tool 4: Bridging Pathway Generator ---
@tool
def recommend_bridging_pathways(current_role: str, target_role: str) -> str:
    """Recommends 3-5 short certifications, CME courses, or upskilling programs to bridge the skill gap between a current role and a target role."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a career advisor for medical professionals."),
        ("user", """A doctor with the role '{current_role}' wants to become a '{target_role}'.
        Recommend 3 to 5 short certifications, CME courses, or upskilling programs they should take to bridge the skill gap.

        Format the output as a numbered list of certification names with short descriptions (not full paragraphs).
        """)
    ])
    chain = prompt | llm
    response = chain.invoke({"current_role": current_role, "target_role": target_role})
    return response.content.strip()

# --- Main Agent Definition ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

def create_medical_mobility_agent():
    tools = [extract_cv_info, detect_medical_role, get_lateral_specialty_recommendations, recommend_bridging_pathways]
    
    # We are using a tool-calling agent, which is suitable for Gemini 1.5 models
    # It allows the LLM to decide which tool to call based on the user's intent.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert medical career advisor AI. Your goal is to help medical professionals identify lateral career opportunities and the pathways to achieve them."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}") # This is where the agent stores its internal thoughts and actions
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor