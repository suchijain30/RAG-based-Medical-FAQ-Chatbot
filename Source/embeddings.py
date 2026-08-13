"""
embeddings.py – Build and save FAISS vector store from cleaned CSV.
Usage: python src/embeddings.py
"""

import os
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "Data", "medical_faqs_clean.csv")
DEFAULT_INDEX = os.path.join(BASE_DIR, "VectorStore", "medical_faq_index")

HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512       # Smaller chunks → better retrieval precision
CHUNK_OVERLAP = 64
BATCH_SIZE = 256       # Embed N chunks at once → fast without OOM


def create_vector_store(
    input_file: str = DEFAULT_INPUT,
    faiss_index_path: str = DEFAULT_INDEX
) -> FAISS:
    os.makedirs(faiss_index_path, exist_ok=True)

    print(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file)

    # Validate columns
    required = {"Question", "Answer"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}")

    df = df.dropna(subset=["Question", "Answer"])
    df["text"] = df["Question"].astype(str) + "\n" + df["Answer"].astype(str)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents: list[Document] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting"):
        chunks = splitter.split_text(row["text"])
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "question": row["Question"],
                    "source": f"{idx}_{i}"
                }
            ))

    print(f"Total chunks: {len(documents)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE}
    )

    # Build FAISS index in one shot (no multiprocessing — avoids pickling errors)
    print("Building FAISS index…")
    texts = [d.page_content for d in documents]
    metas = [d.metadata for d in documents]
    vectordb = FAISS.from_texts(texts, embeddings, metadatas=metas)

    vectordb.save_local(faiss_index_path)
    print(f"Saved FAISS index → {faiss_index_path}")
    return vectordb


if __name__ == "__main__":
    create_vector_store()