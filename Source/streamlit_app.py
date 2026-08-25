"""
streamlit_app.py - Phase 3C: Multilingual + Voice + Image + Login + Full persistent memory
Run: streamlit run Source/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import datetime
from rag_pipeline import (
    load_vector_store, initialize_rag_chain, ask_question_cached,
    transcribe_audio, detect_language
)
from vision import analyze_medical_image, get_mime_type
from auth import signup, login, save_message, load_all_messages, delete_history

FIREBASE_READY = bool(os.getenv("FIREBASE_API_KEY") and os.getenv("FIREBASE_PROJECT_ID"))

st.set_page_config(
    page_title="MediBot – AI Health Assistant",
    page_icon="💊",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f9fc; }
[data-testid="stSidebar"]          { background: #1a1f36; }
[data-testid="stSidebar"] *        { color: white !important; }
[data-testid="stSidebar"] input    { color: #111 !important; }

.medibot-header {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    padding: 1.2rem 2rem; border-radius: 12px;
    color: white; margin-bottom: 1rem;
}
.medibot-header h1 { color: white; margin: 0; font-size: 1.8rem; }
.medibot-header p  { color: #bbdefb; margin: 0.2rem 0 0; font-size: 0.9rem; }

.auth-wrap {
    max-width: 420px; margin: 3rem auto;
    background: white; border-radius: 16px;
    padding: 2.5rem; box-shadow: 0 4px 24px rgba(0,0,0,0.10);
}
.doctor-card {
    background: #e3f2fd; border-left: 4px solid #1565c0;
    border-radius: 8px; padding: 0.7rem 1rem; margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Specialty detection ────────────────────────────────────────────────────
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

def detect_specialty(q: str) -> str:
    q = q.lower()
    for kw, sp in SPECIALTY_MAP.items():
        if kw in q:
            return sp
    return "General Physician"

def detect_mode(t: str) -> str:
    t = t.lower()
    if any(w in t for w in ["summarize","summarise","simplify","explain this"]):
        return "summarize"
    if any(w in t for w in ["generate guide","create report","full overview",
                              "complete guide","everything about"]):
        return "report"
    return "chat"

def practo_url(city, sp):
    return f"https://www.practo.com/{city.lower().replace(' ','-')}/{sp.lower().replace(' ','-')}"

def justdial_url(city, sp):
    return f"https://www.justdial.com/{city.lower().replace(' ','+')}/{sp.lower().replace(' ','+')}"

# ── Session defaults ───────────────────────────────────────────────────────
DEFAULTS = {
    "logged_in": False, "user_id": "", "id_token": "",
    "user_email": "", "city": "", "chat_display": [],
    "agent_executor": None, "history_loaded": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load pipeline factory (global, cached) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading MediBot engine…")
def load_pipeline():
    try:
        vs = load_vector_store()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    return initialize_rag_chain(vs)   # returns create_executor factory

create_executor = load_pipeline()

# ══════════════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div class="medibot-header">
        <h1>💊 MediBot – AI Health Assistant</h1>
        <p>Agentic RAG · Multi-turn Memory · Persistent History</p>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
        st.subheader("🔐 Login or Sign Up")

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        # ── Login tab ─────────────────────────────────────────────────
        with tab_login:
            email_l    = st.text_input("Email",    key="l_email")
            password_l = st.text_input("Password", key="l_pass", type="password")

            if st.button("Login", use_container_width=True, type="primary", key="btn_login"):
                if not email_l or not password_l:
                    st.error("Please enter email and password.")
                elif not FIREBASE_READY:
                    # Dev mode — bypass Firebase
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email_l or "demo@medibot.com"
                    st.session_state.user_id    = "demo_user"
                    st.rerun()
                else:
                    with st.spinner("Logging in…"):
                        res = login(email_l, password_l)
                    if res["ok"]:
                        st.session_state.logged_in  = True
                        st.session_state.id_token   = res["id_token"]
                        st.session_state.user_id    = res["user_id"]
                        st.session_state.user_email = res["email"]
                        st.rerun()
                    else:
                        st.error(res["error"])

        # ── Signup tab ────────────────────────────────────────────────
        with tab_signup:
            email_s    = st.text_input("Email",            key="s_email")
            password_s = st.text_input("Password",         key="s_pass",  type="password")
            confirm_s  = st.text_input("Confirm Password", key="s_conf",  type="password")

            if st.button("Create Account", use_container_width=True,
                         type="primary", key="btn_signup"):
                if not email_s or not password_s:
                    st.error("Please fill all fields.")
                elif password_s != confirm_s:
                    st.error("Passwords do not match.")
                elif len(password_s) < 6:
                    st.error("Password must be at least 6 characters.")
                elif not FIREBASE_READY:
                    st.warning("Firebase not configured. Add keys to .env to enable signup.")
                else:
                    with st.spinner("Creating account…"):
                        res = signup(email_s, password_s)
                    if res["ok"]:
                        st.session_state.logged_in  = True
                        st.session_state.id_token   = res["id_token"]
                        st.session_state.user_id    = res["user_id"]
                        st.session_state.user_email = res["email"]
                        st.rerun()
                    else:
                        st.error(res["error"])

        if not FIREBASE_READY:
            st.info("ℹ️ Firebase not configured — login works in dev mode only.")

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# MAIN APP (after login)
# ══════════════════════════════════════════════════════════════════════════

# Load full history from Firestore on first run after login
if not st.session_state.history_loaded:
    past = []
    if FIREBASE_READY and st.session_state.id_token:
        with st.spinner("Loading your history…"):
            past = load_all_messages(
                st.session_state.user_id,
                st.session_state.id_token
            )
        # Populate chat display (user + assistant pairs)
        st.session_state.chat_display = [
            {"role": m["role"], "content": m["content"],
             "city": m.get("city",""), "specialty": m.get("specialty","")}
            for m in past
        ]

    # Create executor and inject ALL past messages into memory
    st.session_state.agent_executor = create_executor(past_messages=past)
    st.session_state.history_loaded = True


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.user_email}")
    st.caption("✅ Logged in")

    if st.button("🚪 Logout", use_container_width=True):
        for k in DEFAULTS:
            st.session_state[k] = DEFAULTS[k]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📍 Your City")
    city_in = st.text_input("City for doctor links:",
                             placeholder="e.g. Mumbai, Pune",
                             value=st.session_state.city)
    if city_in.strip():
        st.session_state.city = city_in.strip()

    st.markdown("---")
    st.markdown("### ✨ What you can ask")
    st.markdown("""
- 💬 Any medical question
- 🔁 Follow-ups (bot remembers context)
- 🎙️ **Speak** your question (voice input)
- 📷 **Upload** symptom photo / prescription
- 📝 *Summarize this:* [paste text]
- 📋 *Generate a guide for diabetes*
- 🧮 *BMI for 70kg, 1.75m*
""")
    st.markdown("---")

    msg_count = len([m for m in st.session_state.chat_display if m["role"] == "user"])
    st.caption(f"💬 {msg_count} questions in history")

    if st.session_state.chat_display:
        if st.button("🗑️ Clear All History", use_container_width=True):
            if FIREBASE_READY and st.session_state.id_token:
                with st.spinner("Clearing…"):
                    delete_history(st.session_state.user_id, st.session_state.id_token)
            st.session_state.chat_display   = []
            st.session_state.agent_executor = create_executor(past_messages=[])
            st.rerun()

    st.markdown("---")

    # Language indicator — detect from last user message
    last_user_msgs = [m["content"] for m in st.session_state.chat_display if m["role"] == "user"]
    if last_user_msgs:
        detected_lang = detect_language(last_user_msgs[-1])
        lang_emoji = {"Hindi": "🇮🇳", "Marathi": "🇮🇳", "Bengali": "🇮🇳",
                      "Tamil": "🇮🇳", "Telugu": "🇮🇳", "Kannada": "🇮🇳",
                      "Malayalam": "🇮🇳", "Gujarati": "🇮🇳", "Punjabi": "🇮🇳",
                      "English": "🇬🇧"}.get(detected_lang, "🌐")
        st.caption(f"{lang_emoji} Language: {detected_lang}")

    st.warning("⚠️ For informational purposes only.")


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="medibot-header">
    <h1>🩺 MediBot – AI Health Assistant</h1>
    <p>Agentic RAG · Voice & Image · Multilingual · Groq Qwen-3.6</p>
</div>""", unsafe_allow_html=True)

# ── Render chat ────────────────────────────────────────────────────────────
for msg in st.session_state.chat_display:
    with st.chat_message(msg["role"],
                         avatar="💊" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("city") and msg.get("specialty"):
            st.markdown(
                f'<div class="doctor-card">🏥 <b>Find a {msg["specialty"]}</b>'
                f' near you in <b>{msg["city"]}</b></div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔍 Practo",   practo_url(msg["city"], msg["specialty"]), use_container_width=True)
            with c2:
                st.link_button("📋 JustDial", justdial_url(msg["city"], msg["specialty"]), use_container_width=True)

# ── Voice input ──────────────────────────────────────────────────────────────────

with st.expander("🎙️ Voice Input — Speak your question", expanded=False):
    audio_value = st.audio_input(
        "Record your medical question",
        key="voice_input"
    )
    if audio_value is not None:
        # Show playback so user can verify
        st.audio(audio_value, format="audio/wav")

        if st.button("📤 Transcribe & Send", type="primary", key="btn_transcribe"):
            with st.spinner("🎙️ Transcribing your voice..."):
                audio_bytes = audio_value.getvalue()
                transcribed_text = transcribe_audio(audio_bytes)

            if transcribed_text.startswith("⚠️"):
                st.error(transcribed_text)
            else:
                st.success(f'🗒️ Transcribed: "{transcribed_text}"')
                st.session_state["voice_query"] = transcribed_text
                st.rerun()

# Process voice query if it was just transcribed
voice_query = st.session_state.pop("voice_query", None)

# ── Image input ──────────────────────────────────────────────────────────────────

with st.expander("📷 Image Input — Upload a symptom photo, prescription, or lab report", expanded=False):
    uploaded_image = st.file_uploader(
        "Upload a medical image",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_upload",
        help="Supports: skin conditions, prescriptions, lab reports"
    )
    image_question = st.text_input(
        "Optional: Ask something about this image",
        placeholder="e.g. What is this rash? / Read this prescription",
        key="image_question"
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image", width=300)

        if st.button("🔍 Analyze Image", type="primary", key="btn_analyze_img"):
            with st.spinner("📷 Analyzing your image..."):
                img_bytes = uploaded_image.getvalue()
                mime = get_mime_type(uploaded_image.name)
                analysis = analyze_medical_image(
                    img_bytes,
                    user_question=image_question,
                    mime_type=mime
                )

            if analysis.startswith("⚠️"):
                st.error(analysis)
            else:
                # Store analysis to display as chat message
                st.session_state["image_analysis"] = analysis
                st.session_state["image_question_text"] = (
                    image_question if image_question
                    else f"[Uploaded image: {uploaded_image.name}]"
                )
                st.rerun()

# Process image analysis if it was just completed
image_analysis = st.session_state.pop("image_analysis", None)
image_question_text = st.session_state.pop("image_question_text", None)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question, paste text to summarize, or request a patient guide…")

# Voice input takes priority if present
if voice_query and not user_input:
    user_input = voice_query

# ── Handle image analysis result ───────────────────────────────────────────
if image_analysis and image_question_text:
    # Show user message
    with st.chat_message("user"):
        st.markdown(f"📷 {image_question_text}")
    st.session_state.chat_display.append(
        {"role": "user", "content": f"📷 {image_question_text}", "city": "", "specialty": ""}
    )

    # Show analysis result
    with st.chat_message("assistant", avatar="💊"):
        st.markdown(image_analysis)
    st.session_state.chat_display.append(
        {"role": "assistant", "content": image_analysis, "city": "", "specialty": ""}
    )

    # Save to Firestore
    if FIREBASE_READY and st.session_state.id_token:
        save_message(st.session_state.user_id, st.session_state.id_token,
                     "user", f"📷 {image_question_text}")
        save_message(st.session_state.user_id, st.session_state.id_token,
                     "assistant", image_analysis)

if user_input and user_input.strip():
    city      = st.session_state.city
    specialty = detect_specialty(user_input)
    mode      = detect_mode(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_display.append(
        {"role": "user", "content": user_input, "city": "", "specialty": ""}
    )

    # Save user message to Firestore
    if FIREBASE_READY and st.session_state.id_token:
        save_message(st.session_state.user_id, st.session_state.id_token,
                     "user", user_input)

    # Get answer
    spinner_msg = {"summarize": "Summarizing…",
                   "report": "Generating guide…",
                   "chat": "Thinking…"}[mode]

    with st.chat_message("assistant", avatar="💊"):
        with st.spinner(spinner_msg):
            answer, _ = ask_question_cached(st.session_state.agent_executor, user_input)
        st.markdown(answer)

        if city and mode == "chat":
            st.markdown(
                f'<div class="doctor-card">🏥 <b>Find a {specialty}</b>'
                f' near you in <b>{city}</b></div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔍 Practo",   practo_url(city, specialty), use_container_width=True)
            with c2:
                st.link_button("📋 JustDial", justdial_url(city, specialty), use_container_width=True)

    st.session_state.chat_display.append(
        {"role": "assistant", "content": answer,
         "city": city if mode == "chat" else "",
         "specialty": specialty if mode == "chat" else ""}
    )

    # Save assistant message to Firestore
    if FIREBASE_READY and st.session_state.id_token:
        save_message(st.session_state.user_id, st.session_state.id_token,
                     "assistant", answer, city, specialty)