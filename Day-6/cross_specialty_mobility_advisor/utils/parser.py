import fitz  # PyMuPDF

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = " ".join([page.get_text() for page in doc])
    return text
