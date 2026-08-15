"""
rag_pipeline.py - Agentic RAG pipeline with fuzzy matching + tools
"""

import os
import re
from functools import lru_cache
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from groq import RateLimitError, AuthenticationError

# ── Fuzzy matching (no extra library needed) ───────────────────────────────
from difflib import get_close_matches

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "VectorStore", "medical_faq_index")
HF_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 5

# ── Common medical term corrections ───────────────────────────────────────
MEDICAL_TERMS = [
    "dengue", "malaria", "diabetes", "hypertension", "tuberculosis",
    "pneumonia", "asthma", "cancer", "cholera", "typhoid", "hepatitis",
    "arthritis", "psoriasis", "eczema", "migraine", "epilepsy", "alzheimer",
    "parkinson", "schizophrenia", "depression", "anxiety", "obesity",
    "anemia", "leukemia", "lymphoma", "meningitis", "encephalitis",
    "appendicitis", "bronchitis", "gastritis", "colitis", "sinusitis",
    "tonsillitis", "conjunctivitis", "dermatitis", "tendinitis",
    "osteoporosis", "fibromyalgia", "lupus", "sclerosis", "diphtheria",
    "measles", "chickenpox", "smallpox", "polio", "rabies", "tetanus",
    "fever", "cough", "cold", "headache", "nausea", "vomiting", "diarrhea",
    "constipation", "fatigue", "insomnia", "allergy", "infection",
    "inflammation", "fracture", "sprain", "wound", "bleeding", "rash",
    "swelling", "pain", "symptoms", "treatment", "diagnosis", "prevention",
    "vaccine", "medicine", "dosage", "side effects", "surgery", "therapy",
]

def fuzzy_correct(text: str) -> str:
    """
    Correct misspelled medical terms in user input using difflib.
    Works word by word — keeps sentence structure intact.
    """
    words = text.split()
    corrected = []
    for word in words:
        clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(clean) < 4:          # skip short words
            corrected.append(word)
            continue
        matches = get_close_matches(clean, MEDICAL_TERMS, n=1, cutoff=0.75)
        if matches and matches[0] != clean:
            # preserve original casing style
            corrected.append(matches[0])
        else:
            corrected.append(word)
    return " ".join(corrected)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
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
    return FAISS.load_local(
        faiss_index_path,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )


def initialize_rag_chain(vectorstore: FAISS) -> AgentExecutor:
    """
    Build a LangChain Tool-Calling Agent with:
    - Tool 1: Medical FAQ retriever (FAISS)
    - Tool 2: DuckDuckGo web search (fallback for unknown topics)
    - Tool 3: Simple calculator (for BMI, dosage math, etc.)
    """
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
        groq_api_key=GROQ_API_KEY
    )

    # ── Tool 1: FAISS retriever ──────────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="medical_faq_retriever",
        description=(
            "Search the medical FAQ knowledge base for health questions. "
            "Use this FIRST for any medical symptom, disease, treatment, "
            "prevention, or medication question."
        )
    )

    # ── Tool 2: Web search (fallback) ────────────────────────────────────
    web_search = DuckDuckGoSearchRun()
    web_tool = Tool(
        name="web_search",
        func=web_search.run,
        description=(
            "Search the web for medical information NOT found in the FAQ database. "
            "Use only when the retriever returns no relevant results. "
            "Always add a disclaimer that web results should be verified."
        )
    )

    # ── Tool 3: Calculator ───────────────────────────────────────────────
    def safe_calculator(expression: str) -> str:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Invalid expression."
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    calc_tool = Tool(
        name="calculator",
        func=safe_calculator,
        description=(
            "Perform simple math calculations. Useful for BMI calculation "
            "(weight_kg / height_m^2), calorie counting, or medication dosage math. "
            "Input must be a plain math expression like '70 / (1.75 ** 2)'."
        )
    )

    tools = [retriever_tool, web_tool, calc_tool]

    # ── Agent prompt ─────────────────────────────────────────────────────
    system_prompt = """You are MediBot, a responsible and concise AI medical FAQ assistant.

Your behaviour:
1. Always use the medical_faq_retriever tool FIRST for any health question.
2. If retriever returns nothing useful, use web_search as a fallback.
3. Use calculator only for numeric/math requests (BMI, dosage, etc.).
4. Keep answers to 3-5 sentences. Be clear and simple.
5. ALWAYS end medical answers with: "Please consult a healthcare professional for personal advice."
6. If the question is completely non-medical, politely say you only answer medical questions.
7. Never make up information. Only use what tools return.

You have access to these tools: {tool_names}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,          # set True to debug tool calls in terminal
        max_iterations=4,       # prevent infinite loops
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )

    return agent_executor


# ── Cache ─────────────────────────────────────────────────────────────────
_cache: dict = {}


def ask_question_cached(agent_executor: AgentExecutor, question: str) -> tuple[str, list]:
    # Step 1: fuzzy-correct typos
    corrected = fuzzy_correct(question)
    cache_key = corrected.strip().lower()

    if cache_key in _cache:
        return _cache[cache_key]

    try:
        result = agent_executor.invoke({
            "input": corrected,
            "tool_names": "medical_faq_retriever, web_search, calculator"
        })
        answer = result.get("output", "No answer returned.")
        source_docs = []   # agent doesn't return raw docs; retriever used internally

    except RateLimitError:
        answer = (
            "⚠️ Groq rate limit reached. "
            "Please wait ~60 seconds and try again."
        )
        source_docs = []

    except AuthenticationError:
        answer = (
            "⚠️ Invalid Groq API key. "
            "Check your GROQ_API_KEY in the .env file."
        )
        source_docs = []

    except Exception as e:
        answer = f"⚠️ Unexpected error: {str(e)}"
        source_docs = []

    _cache[cache_key] = (answer, source_docs)
    return answer, source_docs