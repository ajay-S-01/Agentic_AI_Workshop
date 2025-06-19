🩺 Cross-Specialty Mobility Advisor
This project implements an AI-powered advisor designed to assist medical professionals in identifying lateral career opportunities and the necessary pathways to transition into new specialties. Leveraging large language models (LLMs) from Google's Gemini, LangChain for agent orchestration, and Retrieval-Augmented Generation (RAG) with ChromaDB, the advisor analyzes a professional's CV and recommends adjacent medical roles and bridging certifications.

🌟 Features
Profile Parsing Agent: Extracts key information (professional summary, education, certifications, experience, skills) from a medical professional's CV (PDF format).
Specialty Proximity Mapper (RAG-Enabled): Identifies the user's current medical role and, using a knowledge base of medical taxonomy and career transition case studies (via RAG), suggests 3-5 adjacent medical specialties.
Lateral Move Recommender: Provides skill overlap scores for suggested roles.
Bridging Pathway Generator: Recommends targeted micro-certifications, CME courses, or clinical exposures required for qualification into new specialties.
Streamlit User Interface: An intuitive web interface for uploading CVs and viewing recommendations.
💡 Technologies Used
Python: Core programming language.
Google Gemini (via google-generativeai and langchain-google-genai): The Large Language Model powering the intelligence of the agents.
LangChain: Framework for developing applications powered by LLMs, specifically for agentic workflows, RAG, and tool orchestration.
ChromaDB (via langchain-chroma): A lightweight vector database used for storing and retrieving medical knowledge for RAG.
PyMuPDF (fitz): For efficient PDF text extraction.
Streamlit: For building the interactive web application.
python-dotenv: For managing API keys and environment variables securely.
🚀 Setup and Installation
Follow these steps to get the project up and running on your local machine.

1. Clone the Repository
Bash

git clone <your-repository-url>
cd cross_speciality_mobility_advisor
2. Prepare requirements.txt
Ensure your requirements.txt file is correctly configured to avoid dependency conflicts. Crucially, ensure the line numpy==2.3.0 is removed or commented out. Also, verify langchain-community is removed and langchain-chroma is added. Using flexible versions for langchain-* and google-* packages often resolves conflicts.

Here's an example of how your requirements.txt should look (ensure it matches your current project's dependencies):

altair==5.5.0
annotated-types==0.7.0
attrs==25.3.0
blinker==1.9.0
cachetools==5.5.2
certifi==2025.6.15
charset-normalizer==3.4.2
click==8.2.1
gitdb==4.0.12
GitPython==3.1.44
google-ai-generativelanguage
google-api-core
google-api-python-client
google-auth
google-auth-httplib2
google-generativeai
googleapis-common-protos
grpcio
grpcio-status
httplib2
idna
Jinja2==3.1.6
jsonschema
jsonschema-specifications
MarkupSafe==3.0.2
narwhals
# numpy==2.3.0  # <--- THIS LINE MUST BE REMOVED OR COMMENTED OUT
packaging==24.2
pandas
pillow==11.2.1
proto-plus
protobuf
pyarrow
pyasn1
pyasn1_modules
pydantic
pydantic_core
pydeck
pyparsing
python-dateutil
python-dotenv==1.1.0
pytz
referencing
requests
rpds-py
rsa
six
smmap
streamlit
tenacity
toml==0.10.2
tornado
tqdm
typing-inspection
typing_extensions
tzdata
uritemplate
urllib3
langchain
# langchain-community  # <--- THIS LINE SHOULD BE REMOVED
langchain-core
langchain-google-genai
langchain-chroma # <--- THIS LINE SHOULD BE ADDED
pypdf==4.2.0
pymupdf==1.24.5
3. Clean and Set Up Virtual Environment
It's highly recommended to start with a clean virtual environment to avoid leftover conflicting packages.

PowerShell

# Deactivate any active virtual environment
deactivate

# Delete the old virtual environment and ChromaDB data (Windows PowerShell)
Remove-Item -Recurse -Force venv
Remove-Item -Recurse -Force .\chroma_db # Delete existing ChromaDB data

# Create a new virtual environment
python -m venv venv

# Activate the new virtual environment
.\venv\Scripts\activate
4. Install Dependencies
Install all the required Python packages from your requirements.txt. Using --no-cache-dir helps prevent issues from cached faulty installations.

PowerShell

pip install --no-cache-dir -r requirements.txt
5. Set Up Google Gemini API Key
Obtain an API key from the Google AI Studio.

Create a file named .env in the root directory of your project (the same directory as app.py).

Add your API key to the .env file like this:

GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
Important: Do not share your API key publicly or commit your .env file to version control.

6. Prepare RAG Data
Create a directory named data in the root of your project:
Bash

mkdir data
Place your medical taxonomy databases and career transition case studies (as .txt or .pdf files) inside this data/ directory. Example files (medical_specialties_taxonomy.txt, career_transition_case_studies.txt) were provided in the project's example data section. The more comprehensive and relevant your data, the better the RAG performance will be.
7. Initialize ChromaDB (Vector Database)
The RAG system requires a vector database. Run the rag_utils.py script once to load your data, embed it, and store it in ChromaDB. This will create a chroma_db directory.

PowerShell

python utils/rag_utils.py
You should see messages indicating that the database is being created or loaded, and that documents are being processed. This step relies on your API key being correct and having network access to Google's embedding models.

▶️ Running the Application
Once everything is set up and utils/rag_utils.py has run successfully, you can run the Streamlit application:

PowerShell

streamlit run app.py
This will open the application in your web browser, typically at http://localhost:8501.

📂 Project Structure
medical-mobility-advisor/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (e.g., GOOGLE_API_KEY)
├── data/                       # Directory for RAG source documents
│   ├── medical_specialties_taxonomy.txt  # Example medical taxonomy data
│   └── career_transition_case_studies.txt # Example career case studies
└── utils/
    ├── agents.py               # Defines LangChain tools and the main agent executor
    └── rag_utils.py            # Handles ChromaDB setup, document loading, and retrieval
📝 How it Works
CV Upload: Users upload a PDF CV via the Streamlit interface.
PDF Parsing: The extract_cv_info tool (a LangChain tool using PyMuPDF and Gemini) extracts structured text from the PDF.
Role Detection: The detect_medical_role tool (another LangChain tool powered by Gemini) identifies the medical professional's current specialty.
Lateral Role Recommendation (RAG-Enabled):
The get_lateral_specialty_recommendations tool is invoked.
This tool queries the chroma_db (our vector store via langchain-chroma) with the current role and CV context to retrieve relevant medical taxonomy and case studies.
The retrieved context is then passed to Gemini along with the main prompt to generate informed recommendations for lateral roles, including skill overlap scores and required bridging certifications.
Bridging Pathway Generation: If needed, the recommend_bridging_pathways tool can be used to provide detailed recommendations for certifications or courses.
Agent Orchestration: The core create_medical_mobility_agent (using create_tool_calling_agent and AgentExecutor) intelligently decides which of these tools to use in what sequence to fulfill the user's request, demonstrating dynamic decision-making by the LLM.
⚠️ Troubleshooting Common Issues
ModuleNotFoundError or Dependency Conflicts (e.g., numpy):
Ensure your requirements.txt does not explicitly pin numpy to a version that conflicts with langchain. Remove numpy==X.X.X from requirements.txt.
Perform a clean installation: deactivate, Remove-Item -Recurse -Force venv, Remove-Item -Recurse -Force .\chroma_db, then create and activate a new venv, and pip install --no-cache-dir -r requirements.txt.
LangChainDeprecationWarning: The class Chroma was deprecated...:
Update your requirements.txt to remove langchain-community and add langchain-chroma.
Update your import statement in utils/rag_utils.py from from langchain_community.vectorstores import Chroma to from langchain_chroma import Chroma.
Crashes during Embedding/ChromaDB initialization (retriever.invoke fails):
Check GOOGLE_API_KEY: Generate a new API key from Google AI Studio and ensure it's correctly placed in your .env file without extra spaces.
Network/Firewall: Verify your internet connection and ensure no firewall or proxy is blocking access to Google's API endpoints.
Model Availability: Though rare, check the Google Cloud Status Dashboard for any service outages.
Run python utils/rag_utils.py directly after deleting chroma_db to get a precise traceback for embedding issues.
