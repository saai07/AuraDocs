import pdfplumber
import pandas as pd
import io
from core.config import MAX_FILE_SIZE_MB


def extract_text_from_file(uploaded_file):
    """Extract text from PDF, TXT, CSV, or Excel files with size validation."""

    # --- File size check ---
    file_bytes = uploaded_file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.")
    uploaded_file.seek(0)

    # --- Parse by extension ---
    name = uploaded_file.name.lower()
    text = ""

    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    elif name.endswith(".txt"):
        text = file_bytes.decode("utf-8")

    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        df = df.fillna("")
        headers = " | ".join(str(col) for col in df.columns)
        text += f"Columns: {headers}\n\n"
        for _, row in df.iterrows():
            text += " | ".join(f"{col}: {val}" for col, val in row.items()) + "\n"

    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
        df = df.fillna("")
        headers = " | ".join(str(col) for col in df.columns)
        text += f"Columns: {headers}\n\n"
        for _, row in df.iterrows():
            text += " | ".join(f"{col}: {val}" for col, val in row.items()) + "\n"

    else:
        raise ValueError(f"Unsupported file type: {name}")

    return text
