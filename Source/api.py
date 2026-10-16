"""
api.py - Phase 4A: FastAPI REST API backend for MediBot
Run: uvicorn Source.api:app --reload --port 8000
"""

import os
import sys
import time
import requests
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

load_dotenv()

# Ensure Source directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    load_vector_store, initialize_rag_chain, ask_question_cached,
    transcribe_audio, detect_language, _get_llm,
)
from vision import analyze_medical_image, get_mime_type
from auth import (
    signup as firebase_signup,
    login as firebase_login,
    verify_id_token,
    FirebaseAuthUnavailable,
    FirestoreRequestError,
    load_profile, save_profile,
    save_message, load_recent_messages, load_all_messages, delete_history,
)


# ── Global State ──────────────────────────────────────────────────────────

# Executor cache: user_id → AgentExecutor (avoids re-creating on every request)
_executor_cache: OrderedDict = OrderedDict()
logger = logging.getLogger(__name__)
_MAX_CACHED_EXECUTORS = 128
_doctor_search_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}
_create_executor = None  # Set in lifespan
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup."""
    global _create_executor
    print("⏳ Loading MediBot pipeline...")
    vs = load_vector_store()
    _create_executor = initialize_rag_chain(vs)
    print("✅ MediBot pipeline loaded!")
    yield
    # Cleanup on shutdown
    _executor_cache.clear()


app = FastAPI(
    title="MediBot API",
    description="AI-Powered Medical FAQ Assistant — REST API",
    version="4.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    ok: bool
    id_token: str = ""
    user_id: str = ""
    email: str = ""
    error: str = ""


class ProfileRequest(BaseModel):
    fullName: str = Field(min_length=1, max_length=100)
    city: str = Field(min_length=1, max_length=200)
    gender: Literal["Male", "Female", "Transgender"]
    dob: date


class DoctorSearchRequest(BaseModel):
    city: str
    specialty: str = ""


class ChatRequest(BaseModel):
    message: str
    city: str = ""


class ChatResponse(BaseModel):
    answer: str
    specialty: str = ""
    language: str = "English"
    corrected_query: str = ""
    doctor_links: dict = {}


class HistoryMessage(BaseModel):
    role: str
    content: str
    city: str = ""
    specialty: str = ""
    timestamp: str = ""


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]
    total: int
    conversation_count: int
    symptom_count: int
    ai_response_count: int


class ImageAnalysisResponse(BaseModel):
    analysis: str
    question: str = ""


class TranscriptionResponse(BaseModel):
    text: str
    is_error: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_loaded: bool
    active_sessions: int


def generate_health_insights(messages: list[dict]) -> str:
    """Generate an insight from the authenticated user's recent messages only."""
    history = [
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in messages[-30:]
        if message.get("role") in ("user", "assistant", "symptom")
    ]
    if not any(message.startswith(("user:", "symptom:")) for message in history):
        return "There is not enough conversation history to identify a personal health pattern yet."

    prompt = (
        "Summarize this user's recent health-related conversation history in 2-4 concise sentences. "
        "Identify recurring symptoms or changes only when the messages support them. "
        "Do not diagnose, infer unreported measurements, or invent facts. If the history has no "
        "clear recurring pattern, say that and suggest what the user could track. End by reminding "
        "them to consult a healthcare professional for personal advice.\n\n"
        + "\n".join(history)[-6000:]
    )
    result = _get_llm().invoke(prompt)
    return result.content.strip()


# ── Specialty Detection ──────────────────────────────────────────────────

SPECIALTY_MAP = {
    "heart": "Cardiologist", "chest pain": "Cardiologist", "blood pressure": "Cardiologist",
    "diabetes": "Endocrinologist", "thyroid": "Endocrinologist", "sugar": "Endocrinologist",
    "skin": "Dermatologist", "rash": "Dermatologist", "acne": "Dermatologist",
    "psoriasis": "Dermatologist", "eczema": "Dermatologist",
    "bone": "Orthopedic", "joint": "Orthopedic", "knee": "Orthopedic", "back pain": "Orthopedic",
    "eye": "Ophthalmologist", "vision": "Ophthalmologist",
    "ear": "ENT Specialist", "throat": "ENT Specialist", "nose": "ENT Specialist",
    "child": "Pediatrician", "baby": "Pediatrician",
    "fever": "General Physician", "cold": "General Physician", "flu": "General Physician",
    "cough": "Pulmonologist", "lung": "Pulmonologist", "breathing": "Pulmonologist",
    "stomach": "Gastroenterologist", "liver": "Gastroenterologist",
    "kidney": "Nephrologist", "urine": "Urologist",
    "mental": "Psychiatrist", "anxiety": "Psychiatrist", "depression": "Psychiatrist",
    "cancer": "Oncologist", "tumor": "Oncologist",
    "brain": "Neurologist", "headache": "Neurologist", "migraine": "Neurologist",
    "teeth": "Dentist", "dental": "Dentist",
    "pregnancy": "Gynecologist", "period": "Gynecologist",
    "dengue": "General Physician", "malaria": "General Physician",
}


def _detect_specialty(q: str) -> str:
    q_lower = q.lower()
    for kw, sp in SPECIALTY_MAP.items():
        if kw in q_lower:
            return sp
    return "General Physician"


def _doctor_links(city: str, specialty: str) -> dict:
    if not city:
        return {}
    c = city.lower().replace(" ", "-")
    s = specialty.lower().replace(" ", "-")
    return {
        "practo": f"https://www.practo.com/{c}/{s}",
        "justdial": f"https://www.justdial.com/{city.replace(' ', '+')}/{specialty.replace(' ', '+')}",
    }


# ── Auth dependency ──────────────────────────────────────────────────────

def _get_auth(
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_id_token: str = Header(default="", alias="X-Id-Token"),
) -> dict:
    """Validate Firebase credentials and use the token's actual account ID."""
    if not x_id_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    try:
        verified_user_id = verify_id_token(x_id_token)
    except FirebaseAuthUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not verified_user_id or (x_user_id and x_user_id != verified_user_id):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    return {"user_id": verified_user_id, "id_token": x_id_token}


def _get_or_create_executor(
    user_id: str,
    id_token: str,
    past_messages: list[dict] | None = None,
):
    """Get cached executor or create new one with user's history."""
    executor = _executor_cache.get(user_id)
    if executor is not None:
        _executor_cache.move_to_end(user_id)
        return executor

    # Use only recent context for the model; the complete history stays in Firestore.
    past = past_messages if past_messages is not None else load_recent_messages(user_id, id_token)
    executor = _create_executor(past_messages=past)
    executor._user_id = user_id
    _executor_cache[user_id] = executor
    if len(_executor_cache) > _MAX_CACHED_EXECUTORS:
        _executor_cache.popitem(last=False)
    return executor


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

# ── Health Check ─────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        version="4.0",
        pipeline_loaded=_create_executor is not None,
        active_sessions=len(_executor_cache),
    )


# ── Auth ─────────────────────────────────────────────────────────────────

@app.post("/api/auth/signup", response_model=AuthResponse, tags=["Auth"])
async def api_signup(req: AuthRequest):
    result = firebase_signup(req.email, req.password)
    return AuthResponse(**result)


@app.post("/api/auth/login", response_model=AuthResponse, tags=["Auth"])
async def api_login(req: AuthRequest):
    result = firebase_login(req.email, req.password)
    return AuthResponse(**result)


@app.get("/api/profile", tags=["Profile"])
async def api_get_profile(auth: dict = Depends(_get_auth)):
    try:
        return load_profile(auth["user_id"], auth["id_token"])
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Unable to reach Firestore while loading your profile.") from exc
    except FirestoreRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.put("/api/profile", tags=["Profile"])
async def api_save_profile(req: ProfileRequest, auth: dict = Depends(_get_auth)):
    profile = req.model_dump()
    try:
        return save_profile(auth["user_id"], auth["id_token"], profile)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Unable to reach Firestore while saving your profile.") from exc
    except FirestoreRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── Chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def api_chat(req: ChatRequest, auth: dict = Depends(_get_auth)):
    user_id = auth["user_id"]
    id_token = auth["id_token"]

    past = load_recent_messages(user_id, id_token)
    executor = _get_or_create_executor(user_id, id_token, past)
    answer, _ = ask_question_cached(
        executor, req.message, user_id=user_id, past_messages=past
    )

    specialty = _detect_specialty(req.message)
    language = detect_language(req.message)
    links = _doctor_links(req.city, specialty) if req.city else {}

    # Save messages to Firestore
    save_message(user_id, id_token, "user", req.message)
    save_message(user_id, id_token, "assistant", answer,
                 city=req.city, specialty=specialty)

    return ChatResponse(
        answer=answer,
        specialty=specialty,
        language=language,
        doctor_links=links,
    )


# ── Voice (Speech-to-Text + Chat) ───────────────────────────────────────

@app.post("/api/chat/voice", response_model=ChatResponse, tags=["Chat"])
async def api_voice_chat(
    audio: UploadFile = File(...),
    city: str = Form(""),
    auth: dict = Depends(_get_auth),
):
    user_id = auth["user_id"]
    id_token = auth["id_token"]

    # Transcribe
    audio_bytes = await audio.read()
    transcribed = transcribe_audio(audio_bytes, filename=audio.filename or "recording.wav")

    if transcribed.startswith("⚠️"):
        raise HTTPException(status_code=400, detail=transcribed)

    # Chat with transcribed text
    past = load_recent_messages(user_id, id_token)
    executor = _get_or_create_executor(user_id, id_token, past)
    answer, _ = ask_question_cached(
        executor, transcribed, user_id=user_id, past_messages=past
    )

    specialty = _detect_specialty(transcribed)
    language = detect_language(transcribed)
    links = _doctor_links(city, specialty) if city else {}

    # Save to Firestore
    save_message(user_id, id_token, "user", transcribed)
    save_message(user_id, id_token, "assistant", answer,
                 city=city, specialty=specialty)

    return ChatResponse(
        answer=answer,
        specialty=specialty,
        language=language,
        corrected_query=transcribed,
        doctor_links=links,
    )


# ── Transcribe Only ─────────────────────────────────────────────────────

@app.post("/api/transcribe", response_model=TranscriptionResponse, tags=["Chat"])
async def api_transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = transcribe_audio(audio_bytes, filename=audio.filename or "recording.wav")
    is_error = text.startswith("⚠️")
    return TranscriptionResponse(text=text, is_error=is_error)


# ── Image Analysis ───────────────────────────────────────────────────────

@app.post("/api/chat/image", response_model=ImageAnalysisResponse, tags=["Chat"])
async def api_image_analysis(
    image: UploadFile = File(...),
    question: str = Form(""),
    auth: dict = Depends(_get_auth),
):
    user_id = auth["user_id"]
    id_token = auth["id_token"]

    img_bytes = await image.read()
    mime = get_mime_type(image.filename or "image.jpg")
    analysis = analyze_medical_image(img_bytes, user_question=question, mime_type=mime)

    if analysis.startswith("⚠️"):
        raise HTTPException(status_code=400, detail=analysis)

    # Save to Firestore
    q_text = question if question else f"[Uploaded image: {image.filename}]"
    save_message(user_id, id_token, "user", f"📷 {q_text}")
    save_message(user_id, id_token, "assistant", analysis)

    return ImageAnalysisResponse(analysis=analysis, question=q_text)


# ── History ──────────────────────────────────────────────────────────────

@app.get("/api/history", response_model=HistoryResponse, tags=["History"])
async def api_get_history(auth: dict = Depends(_get_auth)):
    try:
        messages = load_all_messages(auth["user_id"], auth["id_token"])
    except (requests.RequestException, FirestoreRequestError) as exc:
        logger.exception("Unable to load authenticated user's history from Firestore")
        raise HTTPException(status_code=502, detail="Unable to load your saved history from Firestore.") from exc
    return HistoryResponse(
        messages=[HistoryMessage(**m) for m in messages],
        total=len(messages),
        conversation_count=sum(message["role"] == "user" for message in messages),
        symptom_count=sum(message["role"] == "symptom" for message in messages),
        ai_response_count=sum(message["role"] == "assistant" for message in messages),
    )


@app.delete("/api/history", tags=["History"])
async def api_delete_history(auth: dict = Depends(_get_auth)):
    user_id = auth["user_id"]
    delete_history(user_id, auth["id_token"])

    # Clear cached executor
    _executor_cache.pop(user_id, None)

    return {"ok": True, "message": "History cleared successfully"}


@app.post("/api/symptoms", tags=["Symptoms"])
async def api_save_symptom(req: ChatRequest, auth: dict = Depends(_get_auth)):
    save_message(
        auth["user_id"], auth["id_token"], "symptom", req.message,
        city=req.city,
    )
    return {"ok": True}


@app.get("/api/symptoms/insights", tags=["Symptoms"])
async def api_symptom_insights(auth: dict = Depends(_get_auth)):
    try:
        messages = load_recent_messages(auth["user_id"], auth["id_token"])
    except (requests.RequestException, FirestoreRequestError) as exc:
        logger.exception("Unable to load history for authenticated user's health insight")
        raise HTTPException(status_code=502, detail="Unable to load your saved history for AI analysis.") from exc

    try:
        insight = generate_health_insights(messages)
    except Exception as exc:
        logger.exception("AI health insight generation failed")
        raise HTTPException(
            status_code=503,
            detail="AI analysis is temporarily unavailable. Your saved conversation history is still available.",
        ) from exc
    return {"insight": insight}


@app.post("/api/doctors/search", tags=["Doctors"])
async def api_search_doctors(req: DoctorSearchRequest, auth: dict = Depends(_get_auth)):
    if not TOMTOM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Doctor search is not configured. Set TOMTOM_API_KEY on the backend.",
        )
    if not req.city.strip():
        raise HTTPException(status_code=400, detail="A city is required to search doctors.")

    cache_key = (req.city.strip().casefold(), req.specialty.strip().casefold())
    cached = _doctor_search_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < 600:
        return {"doctors": cached[1]}

    query = f"{req.specialty.strip() + ' ' if req.specialty.strip() else ''}doctors in {req.city.strip()}"
    try:
        response = requests.get(
            f"https://api.tomtom.com/search/2/search/{quote(query, safe='')}.json",
            params={"key": TOMTOM_API_KEY, "limit": 20, "idxSet": "POI"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Doctor directory search failed.") from exc

    doctors = []
    for result in response.json().get("results", []):
        poi = result.get("poi") or {}
        name = poi.get("name", "")
        if not name:
            continue

        address = result.get("address") or {}
        position = result.get("position") or {}
        categories = poi.get("categories") or []
        specialty = categories[0] if categories else "Healthcare provider"
        latitude, longitude = position.get("lat"), position.get("lon")
        map_query = (
            f"{latitude},{longitude}"
            if latitude is not None and longitude is not None
            else f"{name} {address.get('freeformAddress', '')}"
        )
        doctors.append({
            "id": result.get("id") or f"{name}:{map_query}",
            "name": name,
            "address": address.get("freeformAddress", ""),
            "specialty": specialty.replace("_", " ").title(),
            "mapsUrl": (
                "https://www.google.com/maps/search/?api=1&query="
                f"{quote(map_query, safe='')}"
            ),
            "website": poi.get("url", ""),
            "phone": poi.get("phone", ""),
        })
    _doctor_search_cache[cache_key] = (time.monotonic(), doctors)
    if len(_doctor_search_cache) > 256:
        oldest = min(_doctor_search_cache, key=lambda key: _doctor_search_cache[key][0])
        _doctor_search_cache.pop(oldest, None)
    return {"doctors": doctors}


# ── Session Management ───────────────────────────────────────────────────

@app.post("/api/session/refresh", tags=["System"])
async def api_refresh_session(auth: dict = Depends(_get_auth)):
    """Force re-create executor with latest history (e.g. after login)."""
    user_id = auth["user_id"]
    id_token = auth["id_token"]

    # Remove cached executor
    _executor_cache.pop(user_id, None)

    # Re-create with fresh history
    executor = _get_or_create_executor(user_id, id_token)
    past_count = getattr(executor, "_all_past_count", 0)

    return {
        "ok": True,
        "message": "Session refreshed",
        "messages_loaded": past_count,
    }
