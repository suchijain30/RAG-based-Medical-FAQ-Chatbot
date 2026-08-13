import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from groq import RateLimitError, AuthenticationError

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "VectorStore", "medical_faq_index")
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template="""You are a concise, responsible medical FAQ assistant.
Rules:
1. Answer ONLY from the provided context.
2. If context lacks the answer, say: "I don't have enough information. Please consult a healthcare professional."
3. If the question is non-medical, say: "I only answer medical questions."
4. Keep answers to 3-4 sentences max.

Context:
{context}

Question: {input}

Answer:"""
)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vector_store(faiss_index_path: str = FAISS_INDEX_PATH) -> FAISS:
    if not os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_index_path}'.\n"
            "Run: python src/embeddings.py"
        )
    vectordb = FAISS.load_local(
        faiss_index_path,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )
    return vectordb


def initialize_rag_chain(vectorstore: FAISS):
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",   # Best free Groq model — fast + accurate
        temperature=0,
        max_tokens=512,
        groq_api_key=GROQ_API_KEY
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    document_chain = create_stuff_documents_chain(llm=llm, prompt=MEDICAL_PROMPT)
    qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    return qa_chain


_cache: dict = {}


def ask_question_cached(qa_chain, question: str) -> tuple[str, list]:
    key = question.strip().lower()
    if key in _cache:
        return _cache[key]

    try:
        result = qa_chain.invoke({"input": question})
        answer = result.get("answer", "No answer returned.")
        source_docs = result.get("context", [])

    except RateLimitError:
        answer = (
            "⚠️ Groq rate limit reached. You've hit the free tier limit for this minute. "
            "Please wait ~60 seconds and try again. "
            "Limits reset every minute on the free plan."
        )
        source_docs = []

    except AuthenticationError:
        answer = (
            "⚠️ Invalid Groq API key. "
            "Check your GROQ_API_KEY in the .env file. "
            "Get a free key at https://console.groq.com"
        )
        source_docs = []

    except Exception as e:
        answer = f"⚠️ Unexpected error: {str(e)}"
        source_docs = []

    _cache[key] = (answer, source_docs)
    return answer, source_docs