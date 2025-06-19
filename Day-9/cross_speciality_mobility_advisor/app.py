# app.py
import streamlit as st
import os
import json
from utils.agents import create_medical_mobility_agent # Import our agent creator
from utils.rag_utils import setup_vectordb # Import to ensure DB is set up

st.title("🩺 Cross-Specialty Mobility Advisor")

# Initialize the agent once
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_medical_mobility_agent()
    # Ensure ChromaDB is set up on startup
    with st.spinner("Setting up medical knowledge base (ChromaDB)..."):
        db_ready = setup_vectordb()
        if not db_ready:
            st.error("Failed to set up medical knowledge base. RAG functionality might be limited.")
        st.session_state.db_ready = db_ready


uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_pdf_path = "temp_cv.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("Processing CV...")
    st.info("The AI is analyzing your profile and identifying opportunities. This may take a moment.")

    try:
        # Use the agent to extract CV info (Profile Parsing Agent)
        st.subheader("📄 Extracted CV Information:")
        cv_text_response = st.session_state.agent_executor.invoke({"input": f"Extract detailed information from the CV located at {temp_pdf_path}"})
        cv_text = cv_text_response["output"]
        st.text_area("", cv_text[:2000], height=200) # Display a portion of the extracted text

        # Use the agent to detect role (Specialty Proximity Mapper - part 1)
        st.subheader("🧐 Detected Current Role:")
        detected_role_response = st.session_state.agent_executor.invoke({"input": f"What is the medical role of the person based on this CV text: {cv_text[:1000]}?"})
        detected_role = detected_role_response["output"]
        st.success(f"Detected Role: {detected_role}")

        # Use the agent for lateral recommendations (Specialty Proximity Mapper - RAG-Enabled)
        st.subheader("🔁 Suggested Lateral Roles (AI-Powered):")
        # Pass cv_text along for more relevant RAG retrieval
        recommendations_response = st.session_state.agent_executor.invoke(
            {"input": f"Given the current role '{detected_role}' and the CV content: {cv_text[:1000]}, suggest 3-5 lateral or adjacent medical specialties with skill overlap scores and bridging certifications. Format as a JSON list."}
        )
        raw_recommendations_json = recommendations_response["output"]

        try:
            recommendations = json.loads(raw_recommendations_json)
        except json.JSONDecodeError:
            st.error(f"Error parsing recommendations from AI: {raw_recommendations_json}")
            recommendations = [] # Fallback to empty list

        if recommendations:
            for suggestion in recommendations:
                role = suggestion.get("to", "Unknown")
                weight = suggestion.get("weight", "N/A")
                certs = suggestion.get("certifications", [])

                st.markdown(f"- **{role}** (Similarity: {weight})")
                if certs:
                    st.markdown(f"  **Bridging Pathways:**")
                    for cert in certs:
                        st.markdown(f"  - {cert}")
                else:
                    st.markdown("  *No specific bridging certifications recommended directly for this role.*")
                st.markdown("---")

        else:
            st.warning("No lateral role recommendations were generated.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.info("Please ensure your `GOOGLE_API_KEY` is set correctly and the RAG data is available.")

    finally:
        # Clean up temp file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

else:
    st.info("Please upload a PDF CV to get started.")