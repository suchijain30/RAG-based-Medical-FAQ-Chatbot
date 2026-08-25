"""
api.py - Phase 4A: FastAPI REST API backend for MediBot
Run: uvicorn Source.api:app --reload --port 8000
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

load_dotenv()

# Ensure Source directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    load_vector_store, initialize_rag_chain, ask_question_cached,
    transcribe_audio, detect_language, save_lab_report,
)
from vision import analyze_medical_image, get_mime_type
from auth import (
    signup as firebase_signup,
    login as firebase_login,
    save_message, load_all_messages, delete_history,
)


# ── Global State ──────────────────────────────────────────────────────────

# Executor cache: user_id → AgentExecutor (avoids re-creating on every request)
_executor_cache: dict = {}
_create_executor = None  # Set in lifespan


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
    allow_origins=["*"],  # Restrict in production to your domain
    allow_credentials=True,
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
    x_user_id: str = Header(..., alias="X-User-Id"),
    x_id_token: str = Header(..., alias="X-Id-Token"),
) -> dict:
    """Extract user auth from request headers."""
    if not x_user_id or not x_id_token:
        raise HTTPException(status_code=401, detail="Missing authentication headers")
    return {"user_id": x_user_id, "id_token": x_id_token}


def _get_or_create_executor(user_id: str, id_token: str):
    """Get cached executor or create new one with user's history."""
    if user_id in _executor_cache:
        return _executor_cache[user_id]

    # Load history from Firestore and create executor
    past = load_all_messages(user_id, id_token)
    executor = _create_executor(past_messages=past)
    executor._user_id = user_id
    _executor_cache[user_id] = executor
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


# ── Chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def api_chat(req: ChatRequest, auth: dict = Depends(_get_auth)):
    user_id = auth["user_id"]
    id_token = auth["id_token"]

    executor = _get_or_create_executor(user_id, id_token)
    answer, _ = ask_question_cached(executor, req.message, user_id=user_id)

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
    executor = _get_or_create_executor(user_id, id_token)
    answer, _ = ask_question_cached(executor, transcribed, user_id=user_id)

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

    # Save lab report into persistent user memory
    save_lab_report(user_id, analysis)

    # Save to Firestore
    q_text = question if question else f"[Uploaded image: {image.filename}]"
    save_message(user_id, id_token, "user", f"📷 {q_text}")
    save_message(user_id, id_token, "assistant", analysis)

    return ImageAnalysisResponse(analysis=analysis, question=q_text)


# ── History ──────────────────────────────────────────────────────────────

@app.get("/api/history", response_model=HistoryResponse, tags=["History"])
async def api_get_history(auth: dict = Depends(_get_auth)):
    messages = load_all_messages(auth["user_id"], auth["id_token"])
    return HistoryResponse(
        messages=[HistoryMessage(**m) for m in messages],
        total=len(messages),
    )


@app.delete("/api/history", tags=["History"])
async def api_delete_history(auth: dict = Depends(_get_auth)):
    user_id = auth["user_id"]
    delete_history(user_id, auth["id_token"])

    # Clear cached executor
    _executor_cache.pop(user_id, None)

    return {"ok": True, "message": "History cleared successfully"}


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
