import streamlit as st
from utils.parser import extract_text_from_pdf
from utils.llm_graph import generate_specialty_graph, get_fallback_recommendations
from utils.llm_detect import get_detected_role
from utils.llm_cert_recommend import recommend_micro_certifications
import os

st.title("🩺 Cross-Specialty Mobility Advisor")

# Initialize variables
detected_role = None
recommendations = []

uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf("temp.pdf")
    st.subheader("📄 Extracted Text:")
    st.text_area("", text[:2000], height=200)

    # Gemini Role Detection
    try:
        detected_role = get_detected_role(text)
        st.success(f"Detected Role: {detected_role}")
    except Exception as e:
        st.error(f"Error detecting role: {e}")
        detected_role = "General Practitioner"  # Fallback
        st.warning(f"Using fallback role: {detected_role}")

    # Generate dynamic graph
    if detected_role:
        recommendations = generate_specialty_graph(detected_role)
        
        # If API fails, use fallback data
        if not recommendations:
            st.warning("Using fallback recommendations due to API issues...")
            recommendations = get_fallback_recommendations(detected_role)

    # Clean up temp file
    try:
        os.remove("temp.pdf")
    except:
        pass

if recommendations:
    st.subheader("🔁 Suggested Lateral Roles:")
    for suggestion in recommendations:
        role = suggestion.get("to", "Unknown")
        weight = suggestion.get("weight", "?")
        st.markdown(f"- **{role}** (Similarity: {weight})")

    st.subheader("🎓 AI-Powered Micro-Certification Recommendations")
    for suggestion in recommendations:
        role = suggestion.get("to", "Unknown")
        certs = suggestion.get("certifications", [])
        st.markdown(f"**Target Role: {role}**")
        if certs:
            for cert in certs:
                st.markdown(f"- {cert}")
        else:
            st.warning("No certification recommendations found.")
        st.markdown("---")

elif detected_role:
    st.warning("No recommendations were generated. Please check your API key and try again.")
    st.info("💡 Tip: Make sure your GOOGLE_API_KEY is correctly set in your .env file")
else:
    st.info("Please upload a PDF file to get started.")