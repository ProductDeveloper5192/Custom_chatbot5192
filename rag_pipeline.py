import os
import json
import psycopg2
import torch
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
load_dotenv(override=True)

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")

CHUNKS_FILE = "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_MODEL_PATH = r"E:\models\Qwen3-8B"

TABLE_NAME = "document_chunks"
VECTOR_DIM = 384

# =========================
# LOAD MODELS
# =========================
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

tokenizer = None
model = None

def load_qwen():
    global tokenizer, model

    if tokenizer is None or model is None:
        if not os.path.exists(QWEN_MODEL_PATH):
            raise FileNotFoundError(f"Qwen model not found: {QWEN_MODEL_PATH}")

        print("Loading Qwen model...")
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
        ).eval()

# =========================
# DB HELPERS
# =========================
def connect(dbname=None):
    return psycopg2.connect(
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        dbname=dbname or DB_NAME
    )

def create_database():
    conn = connect("postgres")
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    exists = cur.fetchone()

    if not exists:
        cur.execute(f'CREATE DATABASE "{DB_NAME}";')
        print(f"Database '{DB_NAME}' created.")
    else:
        print(f"Database '{DB_NAME}' already exists.")

    cur.close()
    conn.close()

def enable_pgvector():
    conn = connect()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    cur.close()
    conn.close()
    print("pgvector extension enabled.")

def init_table():
    conn = connect()
    register_vector(conn)
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            chunk_id TEXT UNIQUE,
            page_number INT,
            text TEXT,
            embedding VECTOR({VECTOR_DIM})
        );
    """)

    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_embedding
        ON {TABLE_NAME}
        USING hnsw (embedding vector_cosine_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Table initialized.")

def initialize_database():
    create_database()
    enable_pgvector()
    init_table()

# =========================
# STORE CHUNKS
# =========================
def store_embeddings():
    if not os.path.exists(CHUNKS_FILE):
        print(f"{CHUNKS_FILE} not found.")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    conn = connect()
    register_vector(conn)
    cur = conn.cursor()

    cur.execute(f"DELETE FROM {TABLE_NAME};")

    count = 0
    for i, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        chunk_id = str(chunk.get("chunk_id", f"chunk_{i+1}"))
        page_number = int(chunk.get("page_number", 0))
        embedding = embedder.encode(text).tolist()

        cur.execute(f"""
            INSERT INTO {TABLE_NAME} (chunk_id, page_number, text, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING;
        """, (chunk_id, page_number, text, embedding))

        count += 1
        if count % 50 == 0:
            print(f"Inserted {count} chunks...")

    conn.commit()
    cur.close()
    conn.close()

    print(f"Done. Total inserted: {count}")

# =========================
# SEARCH
# =========================
def search_chunks(query, k=3):
    query_embedding = embedder.encode(query).tolist()

    conn = connect()
    register_vector(conn)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT chunk_id, page_number, text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM {TABLE_NAME}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, k))

    rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows

# =========================
# GENERATE ANSWER
# =========================
def ask_question(query):
    rows = search_chunks(query, k=3)

    if not rows:
        return "No context found in database."

    context = "\n\n".join(
        f"Page {page_number}\n{text}"
        for _, page_number, text, _ in rows
    )

    load_qwen()

    prompt = f"""Answer the question only from the context below.

Context:
{context}

Question:
{query}

If the answer is not in the context, say:
"I don't know based on the document."

Answer:
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0
        )

    generated = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

# =========================
# MENU
# =========================
if __name__ == "__main__":
    print("\n1. Initialize Database")
    print("2. Store Embeddings")
    print("3. Ask Question")

    choice = input("Enter choice: ").strip()

    try:
        if choice == "1":
            initialize_database()
        elif choice == "2":
            store_embeddings()
        elif choice == "3":
            question = input("Question: ").strip()
            answer = ask_question(question)
            print("\nAnswer:\n")
            print(answer)
        else:
            print("Invalid choice.")
    except Exception as e:
        print("ERROR:", e)