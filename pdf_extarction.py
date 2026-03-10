from pathlib import Path
import fitz

# =====================================================
# CONFIG
# =====================================================
PDF_PATH = r"E:\RAG_CHAT_BOT\Rag_chat_bot\CoE Kavach Ver 3.2 manual.pdf"        # change to your PDF path
OUTPUT_TEXT_PATH = r"output.txt"   # output text file

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================
def extract_pdf_text(pdf_path):

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = []

    with fitz.open(pdf_file) as doc:

        print(f"Total pages: {len(doc)}")

        for page_num, page in enumerate(doc, start=1):

            print(f"Processing page {page_num}...")

            text = page.get_text("text").strip()

            if text:
                full_text.append(f"\n===== Page {page_num} =====\n")
                full_text.append(text)

    return "\n".join(full_text)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    try:
        extracted_text = extract_pdf_text(PDF_PATH)

        with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print("\nExtraction completed successfully.")
        print(f"Saved file: {OUTPUT_TEXT_PATH}")

    except Exception as e:
        print(f"Error: {e}")