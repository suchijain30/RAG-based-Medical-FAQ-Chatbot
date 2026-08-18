"""
rag_pipeline.py - Phase 2C: Fixed persistent memory + DuckDuckGo rate limit handling
"""

import os
import re
import time
from functools import lru_cache
from difflib import get_close_matches
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from groq import RateLimitError, AuthenticationError

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "VectorStore", "medical_faq_index")
HF_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 5

MEDICAL_TERMS = [
    "dengue","malaria","diabetes","hypertension","tuberculosis","pneumonia",
    "asthma","cancer","cholera","typhoid","hepatitis","arthritis","psoriasis",
    "eczema","migraine","epilepsy","alzheimer","parkinson","schizophrenia",
    "depression","anxiety","obesity","anemia","leukemia","lymphoma",
    "meningitis","encephalitis","appendicitis","bronchitis","gastritis",
    "colitis","sinusitis","tonsillitis","conjunctivitis","dermatitis",
    "osteoporosis","fibromyalgia","lupus","sclerosis","diphtheria","measles",
    "chickenpox","polio","rabies","tetanus","fever","cough","cold","headache",
    "nausea","vomiting","diarrhea","constipation","fatigue","insomnia",
    "allergy","infection","inflammation","fracture","sprain","bleeding","rash",
    "swelling","pain","symptoms","treatment","diagnosis","prevention","vaccine",
    "medicine","dosage","surgery","therapy","monkeypox","coronavirus","covid",
]

def fuzzy_correct(text: str) -> str:
    words = text.split()
    corrected = []
    for word in words:
        clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(clean) < 4:
            corrected.append(word)
            continue
        matches = get_close_matches(clean, MEDICAL_TERMS, n=1, cutoff=0.75)
        corrected.append(matches[0] if matches and matches[0] != clean else word)
    return " ".join(corrected)


FOLLOWUP_PATTERNS = [
    r"\bwhat should i (do|eat|avoid|take|include|exclude)\b",
    r"\bwhich (vegetable|food|fruit|exercise|diet|medicine)\b",
    r"\bcan i (eat|drink|take|do|have)\b",
    r"\bhow (much|many|often|long|should)\b",
    r"\btell me more\b", r"\bwhat else\b",
    r"\bany (tips|advice|suggestion|recommendation)\b",
    r"\bwhat about\b", r"\bshould i\b",
    r"\bis it (safe|ok|okay|good|bad|normal)\b",
    r"\bmore (detail|information|info)\b", r"\bexplain\b",
]

def is_followup(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in FOLLOWUP_PATTERNS)


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
    return FAISS.load_local(
        faiss_index_path,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )


@lru_cache(maxsize=1)
def _get_llm():
    return ChatGroq(
        model="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY
    )


# ── DuckDuckGo with rate limit handling ───────────────────────────────────
def _safe_web_search(query: str) -> str:
    """Web search with retry and graceful fallback on rate limit."""
    try:
        result = DuckDuckGoSearchRun().run(query)
        return result
    except Exception as e:
        err = str(e).lower()
        if "ratelimit" in err or "202" in err or "429" in err:
            # Wait and retry once
            time.sleep(3)
            try:
                return DuckDuckGoSearchRun().run(query)
            except Exception:
                return (
                    "Web search is temporarily unavailable due to rate limiting. "
                    "Based on my medical knowledge base: please consult a healthcare "
                    "professional for the most current treatment information."
                )
        return f"Web search error: {str(e)}"


def _build_tools(vectorstore: FAISS) -> list:

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        ),
        name="medical_faq_retriever",
        description=(
            "Search the medical FAQ knowledge base for symptoms, diseases, "
            "treatments, medications, or prevention. Always call this FIRST."
        )
    )

    web_tool = Tool(
        name="web_search",
        func=_safe_web_search,
        description=(
            "Search the web for medical info not in the FAQ database. "
            "Use ONLY when retriever returns no relevant result."
        )
    )

    def safe_calc(expr: str) -> str:
        try:
            if not all(c in "0123456789+-*/.() " for c in expr):
                return "Invalid expression."
            return f"Result: {eval(expr, {'__builtins__': {}})}"  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    calc_tool = Tool(
        name="calculator",
        func=safe_calc,
        description="Math only — BMI, dosage, calories. Input: plain math like '70/(1.75**2)'."
    )

    def summarize(text: str) -> str:
        try:
            resp = _get_llm().invoke(
                f"Summarize in simple bullet points (max 150 words):\n\n{text}"
            )
            return resp.content
        except Exception as e:
            return f"Error: {e}"

    summarizer_tool = Tool(
        name="summarize_medical_text",
        func=summarize,
        description="Summarize long medical text. Use when user says: summarize/simplify this."
    )

    def generate_guide(condition: str) -> str:
        prompt = (
            f"Write a patient-friendly medical guide for: {condition}\n\n"
            f"Structure:\n"
            f"## What is {condition}?\n"
            f"## Common Symptoms\n## Common Causes\n"
            f"## Treatment Options\n## Prevention Tips\n"
            f"## When to See a Doctor\n\n"
            f"Use bullet points. End with: *Always consult a healthcare professional.*"
        )
        try:
            resp = _get_llm().invoke(prompt)
            return resp.content
        except Exception as e:
            return f"Error: {e}"

    report_tool = Tool(
        name="generate_patient_guide",
        func=generate_guide,
        description=(
            "Generate full patient guide for any condition. "
            "Use when user says: generate guide / full overview / create report."
        )
    )

    return [retriever_tool, web_tool, calc_tool, summarizer_tool, report_tool]


def _build_user_context_summary(past_messages: list[dict]) -> str:
    """
    Build a concise summary of what we know about this user
    from their past conversations — injected as system context.
    This is the KEY fix for cross-session memory.
    """
    if not past_messages:
        return ""

    # Extract user messages only to build context
    user_msgs = [m["content"] for m in past_messages if m["role"] == "user"]
    if not user_msgs:
        return ""

    # Use LLM to summarize what we know about the user
    summary_prompt = (
        "From these past medical conversations, extract key facts about this user "
        "(age, gender, existing conditions, medications, family history, location, "
        "dietary preferences, past symptoms). Be concise, bullet points only, "
        "max 100 words. If nothing personal, say 'No personal context found.':\n\n"
        + "\n".join(user_msgs[-30:])  # last 30 user messages max
    )

    try:
        resp = _get_llm().invoke(summary_prompt)
        summary = resp.content.strip()
        if "no personal context" in summary.lower():
            return ""
        return summary
    except Exception:
        return ""


def initialize_rag_chain(vectorstore: FAISS):
    """Returns factory: create_executor(past_messages=[])"""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\nGet a free key at: https://console.groq.com"
        )

    tools = _build_tools(vectorstore)
    llm   = _get_llm()

    def create_executor(past_messages: list[dict] | None = None) -> AgentExecutor:
        """
        Creates AgentExecutor with full persistent memory.
        past_messages: ALL messages from Firestore for this user.

        Two-layer memory approach:
        1. User context summary → injected in system prompt (who is this user)
        2. Recent messages (last 20) → injected as chat_history (conversation flow)
        """
        past_messages = past_messages or []

        # Layer 1: Build user profile summary from ALL past messages
        user_context = _build_user_context_summary(past_messages)

        # Layer 2: Keep last 20 messages as direct chat history
        recent = past_messages[-20:] if len(past_messages) > 20 else past_messages

        # Build ChatMessageHistory from recent messages
        history = ChatMessageHistory()
        for msg in recent:
            if msg["role"] == "user":
                history.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                history.add_ai_message(msg["content"])

        # System prompt includes user context summary
        system_content = (
            "You are MediBot, a concise and responsible medical FAQ assistant.\n\n"
        )

        if user_context:
            system_content += (
                f"KNOWN USER CONTEXT (from previous sessions):\n{user_context}\n\n"
                "Use this context to personalise your answers. "
                "If user asks 'do you remember me' or 'what do you know about me', "
                "refer to this context and summarise what you know.\n\n"
            )

        system_content += (
            "STRICT RULES:\n"
            "1. Call medical_faq_retriever ONCE for any new medical topic. Then answer.\n"
            "2. If retriever result is empty, call web_search ONCE. Then answer.\n"
            "3. Call calculator only for explicit math (BMI, dosage).\n"
            "4. Call summarize_medical_text only when user says 'summarize'.\n"
            "5. Call generate_patient_guide only when user asks for a full guide/report.\n"
            "6. After ONE tool call, give your final answer immediately.\n"
            "7. Keep answers 3-5 sentences unless generating a guide.\n"
            "8. End medical answers with: 'Please consult a healthcare professional.'\n"
            "9. Never make up information.\n"
            "10. This user's data is private — never mix with other users' info."
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=3,
            max_execution_time=30,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

        # Attach history and metadata to executor
        executor._chat_history    = history
        executor._user_context    = user_context
        executor._all_past_count  = len(past_messages)

        return executor

    return create_executor


def _direct_llm_answer(question: str, history: ChatMessageHistory,
                        user_context: str = "") -> str:
    """Direct LLM call for follow-up questions — no tool needed."""
    system = (
        "You are MediBot, a responsible medical assistant. "
        "Answer the follow-up question using the conversation context. "
        "Give practical, specific advice (diet, exercise, lifestyle). "
        "Keep it clear and under 200 words. "
        "End with: 'Please consult a healthcare professional for personalised advice.'"
    )
    if user_context:
        system += f"\n\nKnown user context:\n{user_context}"

    messages = (
        [SystemMessage(content=system)]
        + history.messages
        + [HumanMessage(content=question)]
    )

    try:
        resp = _get_llm().invoke(messages)
        return resp.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def ask_question_cached(executor: AgentExecutor, question: str) -> tuple[str, list]:
    corrected     = fuzzy_correct(question)
    history       = getattr(executor, "_chat_history",   ChatMessageHistory())
    user_context  = getattr(executor, "_user_context",   "")
    past_count    = getattr(executor, "_all_past_count",  0)

    # Handle "do you remember me" type questions
    memory_triggers = ["remember me", "know about me", "my history",
                       "previous session", "last time", "before"]
    if any(t in corrected.lower() for t in memory_triggers):
        if user_context:
            answer = (
                f"Yes! Based on our previous conversations, here's what I know about you:\n\n"
                f"{user_context}\n\n"
                f"You have {past_count} messages in your history. "
                f"Please consult a healthcare professional for personalised advice."
            )
        else:
            answer = (
                "I have access to your conversation history but haven't found specific "
                "personal details yet. As you share more information, I'll remember it "
                "across all your sessions. Please consult a healthcare professional."
            )
        history.add_user_message(corrected)
        history.add_ai_message(answer)
        return answer, []

    try:
        # Follow-up → direct LLM (no tool call needed)
        if is_followup(corrected) and len(history.messages) > 0:
            answer = _direct_llm_answer(corrected, history, user_context)
        else:
            result = executor.invoke({
                "input": corrected,
                "chat_history": history.messages
            })
            answer = result.get("output", "No answer returned.")

            if not answer or "agent stopped" in answer.lower():
                answer = (
                    "I wasn't able to find a specific answer in my knowledge base. "
                    "Please consult a qualified healthcare professional."
                )

        history.add_user_message(corrected)
        history.add_ai_message(answer)
        return answer, []

    except RateLimitError:
        return "⚠️ Groq rate limit reached. Please wait ~60 seconds and try again.", []
    except AuthenticationError:
        return "⚠️ Invalid GROQ_API_KEY. Check your .env file.", []
    except Exception as e:
        err = str(e)
        if "failed_generation" in err or "function" in err.lower():
            answer = _direct_llm_answer(corrected, history, user_context)
            history.add_user_message(corrected)
            history.add_ai_message(answer)
            return answer, []
        return f"⚠️ Unexpected error: {err}", []