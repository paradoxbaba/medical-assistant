# src/pdf_utils.py

import os
import json
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import logging


# ============================================================
# 0. JSON Helpers for Coursebooks
# ============================================================

BOOK_TRACK_FILE = "ingested_books.json"

def load_ingested_books():
    """Load or initialize ingested_books.json"""
    if not os.path.exists(BOOK_TRACK_FILE):
        return {"Medical_Course": []}
    with open(BOOK_TRACK_FILE, "r") as f:
        return json.load(f)

def save_ingested_books(data):
    """Save updated ingested_books.json"""
    with open(BOOK_TRACK_FILE, "w") as f:
        json.dump(data, f, indent=4)

def book_already_ingested(pdf_name: str) -> bool:
    """Check if coursebook already ingested"""
    data = load_ingested_books()
    return pdf_name in data.get("Medical_Course", [])

def mark_book_ingested(pdf_name: str):
    """Add a new coursebook to ingested_books.json"""
    data = load_ingested_books()
    if pdf_name not in data["Medical_Course"]:
        data["Medical_Course"].append(pdf_name)
        save_ingested_books(data)


# ============================================================
# 1. PDF Loading + Splitting
# ============================================================

def load_pdf_with_fitz(path: str):
    """Extract text blocks from PDF, return LangChain Document objects."""
    docs = []
    pdf = fitz.open(path)

    for i, page in enumerate(pdf):
        blocks = page.get_text("blocks")
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))  # top-to-bottom
        text = "\n".join([b[4] for b in blocks_sorted if b[4].strip()])
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "page": i + 1}
            ))
    return docs

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    """Splits long texts into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# ============================================================
# 2. Upload to Pinecone in Batches
# ============================================================

def upload_in_batches(docs, embedding, index_name, namespace, batch_size=100):
    """Upload documents in batches to avoid 4MB API limit"""
    vector_store = None
    total_batches = (len(docs) - 1) // batch_size + 1
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        batch_size_bytes = sum(len(doc.page_content.encode('utf-8')) for doc in batch)
        print(f"   Batch size: ~{batch_size_bytes/1024/1024:.1f}MB")
        
        try:
            if i == 0:
                vector_store = PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=embedding,
                    index_name=index_name,
                    namespace=namespace
                )
                print(f"✅ Created vector store & uploaded batch {batch_num}")
            else:
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                vector_store.add_texts(texts=texts, metadatas=metadatas)
                print(f"✅ Uploaded batch {batch_num}")
                
        except Exception as e:
            print(f"❌ Error uploading batch {batch_num}: {e}")
            if i == 0:
                print("❌ Critical error: Could not create vector store. Stopping.")
                return None
            continue
    
    return vector_store


# ============================================================
# 3. Coursebook Upload (dedupe via JSON)
# ============================================================

def process_coursebook_pdf(
    pdf_path: str,
    embedding,
    index_name: str,
    batch_size: int = 100,
):
    """
    Ingest a single coursebook PDF into 'Medical_Course' namespace.
    Skips if already ingested.
    """
    filename = os.path.basename(pdf_path)
    if book_already_ingested(filename):
        print(f"⏭️ Skipping {filename}, already ingested in Medical_Course")
        return None

    print(f"📘 Processing coursebook: {filename}")
    docs = load_pdf_with_fitz(pdf_path)
    chunks = split_docs(docs)

    vector_store = upload_in_batches(
        docs=chunks,
        embedding=embedding,
        index_name=index_name,
        namespace="Medical_Course",
        batch_size=batch_size
    )

    if vector_store:
        mark_book_ingested(filename)
        print(f"✅ Uploaded {filename} ({len(chunks)} chunks) "
              f"to Medical_Course namespace")

    return vector_store


# ============================================================
# 4. Patient Upload (overwrite namespace)
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:  # avoid duplicate handlers in Streamlit reloads
    logger.addHandler(handler)


def process_patient_pdf(
    pdf_path: str,
    patient_id: str,
    embedding,
    pc,                # Pinecone client needed for delete
    index_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Ingest (replace) a patient's PDF into its own namespace.
    - Clears old namespace (if exists)
    - Splits PDF into chunks
    - Uploads in one go (patients are usually small)
    """
    if not os.path.exists(pdf_path):
        logger.error(f"❌ PDF not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"📥 Parsing patient PDF: {pdf_path}")
    docs = load_pdf_with_fitz(pdf_path)
    if not docs:
        logger.error(f"❌ No extractable text found in: {pdf_path}")
        raise ValueError(f"No extractable text in: {pdf_path}")

    logger.info(f"✂️ Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})")
    chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"✅ Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")

    index = pc.Index(index_name)
    try:
        index.delete(delete_all=True, namespace=patient_id)
        logger.info(f"🗑️ Cleared old namespace: {patient_id}")
    except Exception:
        logger.warning(f"ℹ️ Namespace {patient_id} not found (first upload). Skipping delete.")

    try:
        logger.info(f"📤 Uploading {len(chunks)} chunks to Pinecone (namespace={patient_id})")
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            index_name=index_name,
            namespace=patient_id
        )
        logger.info(f"✅ Successfully uploaded patient PDF: {os.path.basename(pdf_path)}")
        return vector_store
    except Exception as e:
        logger.exception(f"❌ Error uploading patient PDF {pdf_path}: {e}")
        return None
