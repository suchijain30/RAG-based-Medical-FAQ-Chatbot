# import os
# import pandas as pd
# from tqdm import tqdm
# from dotenv import load_dotenv

# # LangChain imports
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document

# # Load environment variables from .env
# load_dotenv()

# # Smaller HuggingFace model for faster embeddings
# HF_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
# BATCH_SIZE = 64  # Number of chunks per embedding batch
# USE_GPU = False  # Set True if you have GPU with torch + CUDA

# def batch(iterable, n=1):
#     """Yield successive n-sized batches from iterable."""
#     for i in range(0, len(iterable), n):
#         yield iterable[i:i + n]

# def create_vector_store(
#     input_file=r"D:\data_sci_code\RAG_ChatBot\Data\medical_faqs_clean.csv",
#     faiss_index_file=r"D:\data_sci_code\RAG_ChatBot\VectorStore\medical_faq_index"
# ):
#     """
#     Load dataset, split text into chunks, create embeddings in batches,
#     and store them in a FAISS vector store.
#     """

#     # Ensure folder exists
#     os.makedirs(faiss_index_file, exist_ok=True)

#     print(f"Loading dataset from {input_file}...")
#     df = pd.read_csv(input_file)

#     # Combine Question + Answer
#     df["text"] = df["Question"].astype(str) + " " + df["Answer"].astype(str)

#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=200,
#         add_start_index=True
#     )

#     # Split text into chunks and create Document objects
#     documents = []
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting text"):
#         chunks = text_splitter.split_text(row["text"])
#         for i, chunk in enumerate(chunks):
#             documents.append(Document(
#                 page_content=chunk,
#                 metadata={
#                     "question": row["Question"],
#                     "source": f"{idx}_{i}"
#                 }
#             ))

#     print(f"Total chunks created: {len(documents)}")

#     # Initialize embeddings (GPU if available)
#     device = "cuda" if USE_GPU else "cpu"
#     embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME, model_kwargs={"device": device})

#     # Create FAISS vector store in batches
#     print("Creating FAISS vector store in batches...")
#     vectordb = None
#     all_chunks = [doc.page_content for doc in documents]

#     for chunk_batch in tqdm(list(batch(all_chunks, BATCH_SIZE)), desc="Embedding batches"):
#         if vectordb is None:
#             vectordb = FAISS.from_texts(chunk_batch, embeddings)
#         else:
#             vectordb.add_texts(chunk_batch)

#     # Save FAISS index
#     vectordb.save_local(faiss_index_file)
#     print(f"FAISS vector store created and saved at '{faiss_index_file}'")

#     return vectordb


# if __name__ == "__main__":
#     create_vector_store()

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import multiprocessing as mp

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

# HuggingFace model (fast + small)
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
BATCH_SIZE = 128   # bigger batch → fewer forward passes
USE_GPU = False    # True if you have CUDA

# multiprocessing worker
def embed_batch(docs, embeddings):
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)


def create_vector_store(
    input_file=r"D:\data_sci_code\RAG_ChatBot\Data\medical_faqs_clean.csv",
    faiss_index_file=r"D:\data_sci_code\RAG_ChatBot\VectorStore\medical_faq_index"
):
    os.makedirs(faiss_index_file, exist_ok=True)

    print(f"📂 Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)

    # Combine Q + A for retrieval
    df["text"] = df["Question"].astype(str) + " " + df["Answer"].astype(str)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # smaller chunks → faster + more precise retrieval
        chunk_overlap=100,
        add_start_index=True
    )

    # Create documents
    documents = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting text"):
        chunks = splitter.split_text(row["text"])
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "question": row["Question"],
                    "answer": row["Answer"],
                    "source": f"{idx}_{i}"
                }
            ))

    print(f"✅ Total chunks: {len(documents)}")

    # Initialize embeddings
    device = "cuda" if USE_GPU else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": device}
    )

    print("⚡ Embedding in parallel...")
    n_cores = max(1, mp.cpu_count() - 1)  # leave 1 core free
    chunk_size = len(documents) // n_cores + 1
    doc_batches = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]

    with mp.Pool(n_cores) as pool:
        partial_results = pool.starmap(embed_batch, [(batch, embeddings) for batch in doc_batches])

    # Merge FAISS indices
    vectordb = partial_results[0]
    for db in partial_results[1:]:
        vectordb.merge_from(db)

    vectordb.save_local(faiss_index_file)
    print(f"✅ FAISS vector store saved at '{faiss_index_file}'")

    return vectordb


if __name__ == "__main__":
    create_vector_store()
