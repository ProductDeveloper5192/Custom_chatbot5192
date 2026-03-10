from pathlib import Path
import fitz
import pytesseract
from PIL import Image
import io
import json
import chromadb
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = r"Rag_chat_bot\CoE Kavach Ver 3.2 manual.pdf"
JSON_OUTPUT_PATH = r"chunks.json"
TXT_OUTPUT_PATH = r"chromDBexatrcted.txt"
IMAGE_OUTPUT_DIR = r"extracted_images"

MIN_TEXT_LENGTH = 30
OCR_DPI = 300
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Optional for Windows
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.expanduser("~"), r"scoop\shims\tesseract.exe")
os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.expanduser("~"), r"scoop\apps\tesseract\current\tessdata")

def clean_extracted_text(text: str) -> str:
    # Remove the repeating manual footer
    text = re.sub(r'LOCO PILOT OPERATING MANUAL FOR KAVACH SPEC VER 3\.2\s*\|\s*\d+', '', text, flags=re.IGNORECASE)
    
    # Remove large tables of repetitive header noise like "Date of Revision", "Approval Ver", "Prepared By"
    text = re.sub(r'(?si)(Date of Revision.*?Approval Ver.*?Prepared By.*?Reviewed By.*?Checked By.*?Approved By.*?Signature.*?Date.*?\d{4}\.\d{2}\.\d{2}.*?\+05\'30\')', '', text)
    text = re.sub(r'(?si)(Digitally signed by.*?Date:.*?\+05\'30\')', '', text)
    
    # Remove standard header/footer artifacts
    text = re.sub(r'भारतसरकार- रेल मं\S+ Govt\. of India – Ministry of Railways.*?(?=XVIII)', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'भारतीय रेल िसगनल इंजीिनयरी और दूरसंचार सं\S+.*?(?=XVIII|DRAFT)', '', text, flags=re.IGNORECASE|re.DOTALL)

    # Remove strict leading list indices like "11.", "25." etc. but keep Roman numerals intact if they are headers
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Replace literal manual placeholders like X.YY, XXXX, YYYY with a cleaner generic marker
    text = re.sub(r'X\.YY\(\s*X\.YY\)', '[value]', text)
    text = re.sub(r'X\.YY', '[value]', text)
    text = re.sub(r'[XY]{3,}', '[value]', text)
    
    # Remove loose "Page X" references
    text = re.sub(r'\bPage \d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove excessive empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_direct_text(page) -> str:
    # Use 'dict' blocks to get better structured text with newlines
    blocks = page.get_text("dict").get("blocks", [])
    text_content = []
    for block in blocks:
        if block.get('type') == 0:  # text
            block_lines = []
            for line in block.get("lines", []):
                line_text = "".join([span.get("text", "") for span in line.get("spans", [])]).strip()
                if line_text:
                    block_lines.append(line_text)
                    
            if block_lines:
                # Join lines within a block with a space to form a cohesive paragraph
                text_content.append(" ".join(block_lines))
                text_content.append("\n\n") # Double newline between distinct paragraphs
                
    return "".join(text_content).strip()

def extract_ocr_text(page, dpi=300) -> str:
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(image).strip()

def extract_images_from_page(doc, page, page_num, output_dir):
    image_paths = []
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        xref = img[0]
        width, height = img[2], img[3]
        
        # Filter out small graphics, logos, and masks which often render as solid colors
        if width < 150 or height < 150:
            continue
            
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"page_{page_num}_img_{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            # Use forward slashes or consistent slashes for DB
            image_paths.append(image_path.replace("\\", "/"))
        except Exception as e:
            print(f"Failed to extract image on page {page_num}: {e}")
    return image_paths

def extract_and_chunk_pdf(pdf_path: str):
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    all_chunks = []
    
    # Semantic text splitter keeps sentences/paragraphs intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    with fitz.open(pdf_file) as doc:
        for page_num, page in enumerate(doc, start=1):
            direct_text = extract_direct_text(page)

            if len(direct_text) >= MIN_TEXT_LENGTH:
                page_text = direct_text
                method = "DIRECT_TEXT"
            else:
                page_text = extract_ocr_text(page, dpi=OCR_DPI)
                method = "OCR"
                
            # Clean the text before chunking
            page_text = clean_extracted_text(page_text)
                
            # Extract images for this page
            image_paths = extract_images_from_page(doc, page, page_num, IMAGE_OUTPUT_DIR)
            images_str = ",".join(image_paths)

            # Use Langchain Text Splitter
            page_chunks = text_splitter.split_text(page_text)

            for idx, chunk in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "page_number": page_num,
                    "chunk_id": f"page_{page_num}_chunk_{idx}",
                    "method": method,
                    "images": images_str,
                    "text": chunk
                })

    return all_chunks

if __name__ == "__main__":
    try:
        chunks = extract_and_chunk_pdf(PDF_PATH)

        with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        with open(TXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(str(chunk["text"]) + "\n\n")

        print(f"Chunked output saved to: {JSON_OUTPUT_PATH}")
        print(f"Text output saved to: {TXT_OUTPUT_PATH}")
        print(f"Total chunks: {len(chunks)}")

        print("\nInitializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="doc_chunks_v7")
        
        # Prepare data for Chroma DB
        documents = []
        metadatas = []
        ids = []
        
        for idx, c in enumerate(chunks):
            documents.append(c["text"])
            
            # Make sure metadata does not contain complex types, only string/int/float
            metadatas.append({
                "page_number": c["page_number"], 
                "method": c["method"],
                "images": c["images"]
            })
            
            # Ensure unique IDs just in case
            ids.append(f"{c['chunk_id']}")
            
        if documents:
            print("Adding chunks to Chroma DB... (this might take a moment to generate embeddings)")
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(documents)} chunks to the 'doc_chunks_v7' collection in ChromaDB.")

    except Exception as e:
        print(f"Error: {e}")



        