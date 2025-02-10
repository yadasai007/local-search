from PyPDF2 import PdfReader
import os
import glob
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"  # Extract text from each page
    return text.strip()

def extract_text_from_all_pdfs(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))  # Get all PDF files
    pdf_texts = {}

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        pdf_texts[os.path.basename(pdf_file)] = text  # Store text with filename

    return pdf_texts