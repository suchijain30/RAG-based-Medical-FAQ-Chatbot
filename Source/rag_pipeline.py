"""
rag_pipeline.py - Phase 3C: Multilingual + Voice + Image + Persistent Memory + Token Budgeting
"""

import os
import re
import json
import time
import tempfile
from pathlib import Path
from functools import lru_cache
from difflib import get_close_matches
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from groq import Groq, RateLimitError, AuthenticationError

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
BASE_DIR         = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX", str(BASE_DIR / "VectorStore" / "medical_faq_index"))
USER_MEMORY_DIR  = BASE_DIR / "user_memory"
HF_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 6

# Ensure user_memory directory exists
USER_MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ── Persistent JSON User Memory ─────────────────────────────────────────────

def load_user_memory(user_id: str = "default_user") -> dict:
    """
    Load persistent user memory from user_memory/<user_id>.json.
    Returns default schema if not found or corrupted.
    """
    user_file = USER_MEMORY_DIR / f"{user_id}.json"
    default_memory = {
        "user_id": user_id,
        "profile": {
            "age": None,
            "gender": None,
            "conditions": []
        },
        "lab_report": "",
        "history": []
    }
    if not user_file.exists():
        return default_memory

    try:
        with open(user_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return default_memory
            if "profile" not in data or not isinstance(data["profile"], dict):
                data["profile"] = {"age": None, "gender": None, "conditions": []}
            if "conditions" not in data["profile"] or not isinstance(data["profile"]["conditions"], list):
                data["profile"]["conditions"] = []
            if "lab_report" not in data:
                data["lab_report"] = ""
            if "history" not in data or not isinstance(data["history"], list):
                data["history"] = []
            data["user_id"] = user_id
            return data
    except Exception:
        return default_memory


def save_user_memory(user_id: str, memory_data: dict) -> None:
    """Save persistent user memory to user_memory/<user_id>.json."""
    USER_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    user_file = USER_MEMORY_DIR / f"{user_id}.json"
    try:
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save user memory for {user_id}: {e}")


def save_lab_report(user_id: str, report_text: str) -> None:
    """Helper to store uploaded lab report analysis in user memory."""
    memory = load_user_memory(user_id)
    memory["lab_report"] = report_text.strip()
    save_user_memory(user_id, memory)


def extract_profile_from_text(text: str, current_profile: dict | None = None) -> dict:
    """
    Extract profile facts (age, gender, conditions) from user text without a separate LLM call.
    Updates current_profile in-place and returns it.
    """
    if current_profile is None:
        current_profile = {"age": None, "gender": None, "conditions": []}

    t_lower = text.lower()

    # 1. Age extraction
    age_patterns = [
        r"\b(?:i am|i'm|age is|age:?|aged)\s*(?:a\s+)?(\d{1,3})\s*(?:years?|yrs?|yo)?\b",
        r"\b(\d{1,3})\s*(?:years?[- ]old|yrs?[- ]old|yo)\b",
        r"\b(\d{1,2})\s*years?\s+of\s+age\b",
    ]
    for pat in age_patterns:
        match = re.search(pat, t_lower)
        if match:
            try:
                val = int(match.group(1))
                if 0 < val < 120:
                    current_profile["age"] = val
                    break
            except (ValueError, IndexError):
                pass

    # 2. Gender extraction
    gender_patterns = [
        (r"\b(?:i am|i'm)\s+(?:a\s+)?(male|man|boy|gentleman)\b", "Male"),
        (r"\b(?:i am|i'm)\s+(?:a\s+)?(female|woman|girl|lady)\b", "Female"),
        (r"\b\d{1,3}\s*(?:years?[- ]old|yrs?[- ]old)?\s*(male|man|boy)\b", "Male"),
        (r"\b\d{1,3}\s*(?:years?[- ]old|yrs?[- ]old)?\s*(female|woman|girl|lady)\b", "Female"),
        (r"\b(?:gender|sex)[:\s]+(male|man)\b", "Male"),
        (r"\b(?:gender|sex)[:\s]+(female|woman)\b", "Female"),
    ]
    for pat, g_val in gender_patterns:
        if re.search(pat, t_lower):
            current_profile["gender"] = g_val
            break

    # 3. Medical conditions extraction
    known_conditions = [
        "diabetes", "hypertension", "high blood pressure", "low blood pressure",
        "asthma", "arthritis", "migraine", "thyroid", "hypothyroidism", "hyperthyroidism",
        "cholesterol", "high cholesterol", "eczema", "psoriasis", "depression",
        "anxiety", "anemia", "cancer", "gerd", "acid reflux", "obesity", "epilepsy",
        "tuberculosis", "pneumonia", "bronchitis", "fatty liver", "kidney stone",
        "osteoporosis", "pcos", "pcod", "sinusitis", "tonsillitis"
    ]

    existing = set(c.title() for c in current_profile.get("conditions", []))

    for cond in known_conditions:
        if re.search(rf"\b{re.escape(cond)}\b", t_lower):
            existing.add(cond.title())

    condition_phrases = [
        r"\b(?:i have|i've got|diagnosed with|suffering from|living with|history of|treatment for|medication for)\s+([a-zA-Z\s,]+)",
    ]
    for pat in condition_phrases:
        matches = re.finditer(pat, t_lower)
        for m in matches:
            phrase = m.group(1).strip()
            for med in MEDICAL_TERMS:
                if len(med) >= 4 and re.search(rf"\b{re.escape(med)}\b", phrase):
                    existing.add(med.title())

    current_profile["conditions"] = sorted(list(existing))
    return current_profile


# ── History Token Budgeting ────────────────────────────────────────────────

def _trim_history(history: list[dict], max_tokens: int = 1800) -> list[dict]:
    """
    Trim conversation history to fit within max_tokens budget (1 token ≈ 4 chars).
    Walks backward through history and cuts off older messages when budget is exceeded.
    Always preserves a minimum of 2 conversation turns (up to 4 messages if available)
    so context never completely vanishes.
    """
    if not history:
        return []

    max_chars = max_tokens * 4
    min_messages = min(len(history), 4)  # Minimum 2 turns (4 messages)

    selected: list[dict] = []
    current_chars = 0

    # Walk backward from most recent to oldest
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        content_len = len(msg.get("content", ""))

        if len(selected) < min_messages or (current_chars + content_len <= max_chars):
            selected.append(msg)
            current_chars += content_len
        else:
            break

    # Restore chronological order
    selected.reverse()
    return selected


# ── Voice Transcription (Groq Whisper) ─────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str = "recording.wav") -> str:
    """
    Transcribe audio bytes to text using Groq Whisper API.
    Returns transcribed text or error message.
    """
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set. Cannot transcribe audio."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            prompt=(
                "Medical consultation transcript. Common terms: "
                "diabetes, hypertension, dengue, malaria, tuberculosis, "
                "pneumonia, asthma, migraine, arthritis, cholesterol, "
                "thyroid, insulin, paracetamol, ibuprofen, metformin."
            ),
            response_format="text",
            language="en",
            temperature=0.0,
        )
        text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        if not text:
            return "⚠️ Could not detect any speech. Please try again."
        return text
    except RateLimitError:
        return "⚠️ Whisper rate limit reached. Please wait a moment and try again."
    except Exception as e:
        err = str(e)
        if "audio" in err.lower() or "format" in err.lower():
            return "⚠️ Unsupported audio format. Please record again."
        return f"⚠️ Transcription error: {err}"


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


REPORT_FOLLOWUP_PATTERNS = [
    r"\bwhat should i (?:eat|avoid|drink|take|do|include|exclude)\b",
    r"\bwhich (?:food|diet|fruit|vegetable|exercise|medicine|supplement)\b",
    r"\bcan i (?:eat|drink|have|take|exercise)\b",
    r"\b(?:diet|nutrition|food|meals?|exercise|workout|routine|lifestyle)\b",
    r"\b(?:my report|the report|my results?|test results?|lab results?|blood test|values?|levels?|readings?|numbers?)\b",
    r"\b(?:is (?:it|this) normal|are (?:these|my) (?:levels|values|numbers|results)|what do (?:these|the) (?:numbers|results|values) mean)\b",
    r"\b(?:advice|recommendations?|suggestions?|tips|precautions?)\b",
    r"\b(?:based on (?:my|the) (?:report|results|tests?|values))\b",
    r"\bwhat does (?:it|this) mean\b",
    r"\bwhat to do next\b",
]

def _is_followup_about_report(question: str) -> bool:
    """Detect if the question is asking about recommendations/diet/lifestyle or previous lab report."""
    q = question.lower()
    return any(re.search(p, q) for p in REPORT_FOLLOWUP_PATTERNS)


# ── Language Detection ────────────────────────────────────────────────────

_SCRIPT_RANGES = {
    "Hindi":    (0x0900, 0x097F),   # Devanagari
    "Bengali":  (0x0980, 0x09FF),
    "Tamil":    (0x0B80, 0x0BFF),
    "Telugu":   (0x0C00, 0x0C7F),
    "Kannada":  (0x0C80, 0x0CFF),
    "Malayalam":(0x0D00, 0x0D7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Punjabi":  (0x0A00, 0x0A7F),   # Gurmukhi
    "Odia":     (0x0B00, 0x0B7F),
    "Marathi":  (0x0900, 0x097F),   # Uses Devanagari — disambiguated by keywords
    "Arabic":   (0x0600, 0x06FF),   # Urdu uses Arabic script
}

_MARATHI_MARKERS = [
    "आहे", "मी", "तु", "त्या", "काय", "आहेत",
    "मला", "नाही", "तुम्हाला", "करा",
]

def detect_language(text: str) -> str:
    """Detect language from text using Unicode script analysis. Returns language name."""
    script_counts = {}
    for char in text:
        cp = ord(char)
        for lang, (start, end) in _SCRIPT_RANGES.items():
            if start <= cp <= end:
                script_counts[lang] = script_counts.get(lang, 0) + 1

    if not script_counts:
        return "English"

    detected = max(script_counts, key=script_counts.get)

    if detected in ("Hindi", "Marathi"):
        if any(m in text for m in _MARATHI_MARKERS):
            return "Marathi"
        return "Hindi"

    return detected


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
            "Run: python Source/embeddings.py"
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
    """Extract summary from past messages if available."""
    if not past_messages:
        return ""

    user_msgs = [m["content"] for m in past_messages if m.get("role") == "user"]
    if not user_msgs:
        return ""

    summary_prompt = (
        "From these past medical conversations, extract key facts about this user "
        "(age, gender, existing conditions, medications, family history, location, "
        "dietary preferences, past symptoms). Be concise, bullet points only, "
        "max 100 words. If nothing personal, say 'No personal context found.':\n\n"
        + "\n".join(user_msgs[-30:])
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
    """Returns factory: create_executor(past_messages=[], user_profile={}, lab_report='')"""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\nGet a free key at: https://console.groq.com"
        )

    tools = _build_tools(vectorstore)
    llm   = _get_llm()

    def create_executor(past_messages: list[dict] | None = None,
                        user_profile: dict | None = None,
                        lab_report: str = "") -> AgentExecutor:
        past_messages = past_messages or []
        user_profile = user_profile or {}

        # Layer 1: Build profile summary
        profile_parts = []
        if user_profile.get("age"):
            profile_parts.append(f"Age: {user_profile['age']}")
        if user_profile.get("gender"):
            profile_parts.append(f"Gender: {user_profile['gender']}")
        if user_profile.get("conditions"):
            profile_parts.append(f"Conditions: {', '.join(user_profile['conditions'])}")

        user_context = "; ".join(profile_parts) if profile_parts else ""

        # Layer 2: Trim history to 1800 token budget
        trimmed = _trim_history(past_messages, max_tokens=1800)
        history = ChatMessageHistory()
        for msg in trimmed:
            if msg.get("role") == "user":
                history.add_user_message(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                history.add_ai_message(msg.get("content", ""))

        system_content = (
            "You are MediBot, a concise and responsible medical FAQ assistant.\n\n"
        )

        if user_context:
            system_content += (
                f"KNOWN USER PROFILE:\n{user_context}\n\n"
                "Use this profile to personalize your answers. "
                "If user asks 'do you remember me' or 'what do you know about me', "
                "refer to this profile and summarize what you know.\n\n"
            )

        if lab_report:
            system_content += (
                f"LATEST USER LAB REPORT / MEDICAL OBSERVATION:\n{lab_report}\n\n"
                "When the user asks follow-up questions about diet, lifestyle, what to eat, "
                "or next steps, refer to this lab report to give context-aware recommendations.\n\n"
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
            "10. This user's data is private — never mix with other users' info.\n"
            "11. MULTILINGUAL: If the user writes in a non-English language "
            "(Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, "
            "Punjabi, Odia, Urdu, or any other language), you MUST respond in that "
            "same language. Keep medical terms in English but explain in their language."
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

        executor._chat_history    = history
        executor._user_context    = user_context
        executor._lab_report      = lab_report
        executor._all_past_count  = len(past_messages)

        return executor

    return create_executor


def _direct_llm_answer(question: str, history_messages: list,
                        user_context: str = "", lab_report: str = "") -> str:
    """Direct LLM call for follow-up or report questions — fast and context-rich."""
    system = (
        "You are MediBot, a responsible and knowledgeable medical assistant. "
        "Answer the follow-up question using the conversation context and any available medical reports. "
        "Give practical, specific advice (diet, exercise, lifestyle, precautions). "
        "Keep it clear and under 200 words. "
        "End with: 'Please consult a healthcare professional for personalized advice.'"
    )
    if user_context:
        system += f"\n\nKnown user profile:\n{user_context}"
    if lab_report:
        system += f"\n\nUser's uploaded lab report / test data:\n{lab_report}"

    if isinstance(history_messages, ChatMessageHistory):
        msg_list = history_messages.messages
    elif isinstance(history_messages, list):
        msg_list = history_messages
    else:
        msg_list = []

    messages = (
        [SystemMessage(content=system)]
        + msg_list
        + [HumanMessage(content=question)]
    )

    try:
        resp = _get_llm().invoke(messages)
        return resp.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def ask_question(question: str, user_id: str = "default_user", executor: AgentExecutor = None) -> tuple[str, list]:
    """
    Main query function with persistent JSON memory, lab report context injection,
    history trimming (1800 token budget), and multi-turn chat history.
    """
    if executor is not None and callable(executor) and not hasattr(executor, "invoke"):
        try:
            executor = executor()
        except Exception:
            pass

    corrected = fuzzy_correct(question)
    memory = load_user_memory(user_id)

    # 1. Auto-extract profile facts (age, gender, conditions)
    extract_profile_from_text(question, memory["profile"])

    # 2. Check for lab report context injection
    lab_report = memory.get("lab_report", "")
    is_report_q = bool(lab_report and _is_followup_about_report(question))

    # Build profile context
    profile_items = []
    if memory["profile"].get("age"):
        profile_items.append(f"Age: {memory['profile']['age']}")
    if memory["profile"].get("gender"):
        profile_items.append(f"Gender: {memory['profile']['gender']}")
    if memory["profile"].get("conditions"):
        profile_items.append(f"Conditions: {', '.join(memory['profile']['conditions'])}")
    user_context = "; ".join(profile_items)

    # 3. Trim history to 1800 token budget (min 2 turns)
    trimmed_history = _trim_history(memory.get("history", []), max_tokens=1800)

    # Convert trimmed history to LangChain message objects
    history_messages = []
    for msg in trimmed_history:
        if msg.get("role") == "user":
            history_messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            history_messages.append(AIMessage(content=msg.get("content", "")))

    # Handle memory inquiry triggers
    memory_triggers = ["remember me", "know about me", "my history", "previous session", "last time", "before"]
    if any(t in corrected.lower() for t in memory_triggers):
        past_count = len(memory.get("history", []))
        if user_context or lab_report:
            details = []
            if user_context:
                details.append(f"• Profile: {user_context}")
            if lab_report:
                details.append(f"• Lab Report / Test on file: {lab_report[:120]}...")
            answer = (
                f"Yes! Based on your saved profile and previous conversations:\n\n"
                + "\n".join(details) +
                f"\n\nYou have {past_count} messages in your history. "
                "Please consult a healthcare professional for personalized advice."
            )
        else:
            answer = (
                "I have access to your conversation history but haven't recorded specific "
                "personal profile details yet. As you share your age, conditions, or reports, "
                "I will remember them across sessions. Please consult a healthcare professional."
            )
        memory["history"].append({"role": "user", "content": question})
        memory["history"].append({"role": "assistant", "content": answer})
        save_user_memory(user_id, memory)
        return answer, []

    effective_input = corrected
    if is_report_q:
        effective_input = (
            f"[CONTEXT: The user has uploaded the following medical report/test results: {lab_report}]\n\n"
            f"User Question: {corrected}"
        )

    try:
        if (is_followup(corrected) or is_report_q) and len(history_messages) > 0:
            answer = _direct_llm_answer(effective_input, history_messages, user_context, lab_report)
        elif executor is not None and hasattr(executor, "invoke"):
            result = executor.invoke({
                "input": effective_input,
                "chat_history": history_messages
            })
            answer = result.get("output", "No answer returned.")
            if not answer or "agent stopped" in answer.lower():
                answer = (
                    "I wasn't able to find a specific answer in my knowledge base. "
                    "Please consult a qualified healthcare professional."
                )
        else:
            answer = _direct_llm_answer(effective_input, history_messages, user_context, lab_report)

        memory["history"].append({"role": "user", "content": question})
        memory["history"].append({"role": "assistant", "content": answer})
        save_user_memory(user_id, memory)
        return answer, []

    except RateLimitError:
        return "⚠️ Groq rate limit reached. Please wait ~60 seconds and try again.", []
    except AuthenticationError:
        return "⚠️ Invalid GROQ_API_KEY. Check your .env file.", []
    except Exception as e:
        err = str(e)
        if "failed_generation" in err or "function" in err.lower():
            answer = _direct_llm_answer(effective_input, history_messages, user_context, lab_report)
            memory["history"].append({"role": "user", "content": question})
            memory["history"].append({"role": "assistant", "content": answer})
            save_user_memory(user_id, memory)
            return answer, []
        return f"⚠️ Unexpected error: {err}", []


def ask_question_cached(executor, question: str, user_id: str = "default_user") -> tuple[str, list]:
    """Backward-compatible wrapper around ask_question."""
    if hasattr(executor, "_user_id") and getattr(executor, "_user_id"):
        user_id = getattr(executor, "_user_id")
    return ask_question(question=question, user_id=user_id, executor=executor)