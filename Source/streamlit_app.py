"""
streamlit_app.py - MediBot with Firebase Auth + Firestore history + GPS location
Run: streamlit run src/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import datetime
import json
import requests
from streamlit.components.v1 import html as st_html
from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached

# ── Firebase config ────────────────────────────────────────────────────────
# Add these to your .env and Streamlit Cloud secrets
FIREBASE_API_KEY        = os.getenv("FIREBASE_API_KEY", "")
FIREBASE_PROJECT_ID     = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_AUTH_DOMAIN    = os.getenv("FIREBASE_AUTH_DOMAIN", "")

st.set_page_config(
    page_title="MediBot – AI Health Assistant",
    page_icon="💊",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
[data-testid="stAppViewContainer"] { background: #f7f9fc; }
[data-testid="stSidebar"] { background: #1a1f36; color: white; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] input { color: #111 !important; }

/* Header strip */
.medibot-header {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.medibot-header h1 { color: white; margin: 0; font-size: 2rem; }
.medibot-header p  { color: #bbdefb; margin: 0.3rem 0 0; font-size: 0.95rem; }

/* Auth card */
.auth-card {
    background: white;
    border-radius: 16px;
    padding: 2.5rem;
    max-width: 420px;
    margin: 3rem auto;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
}

/* Doctor card */
.doctor-card {
    background: #e3f2fd;
    border-left: 4px solid #1565c0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Specialty map ──────────────────────────────────────────────────────────
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
    "stomach": "Gastroenterologist", "liver": "Gastroenterologist", "digestion": "Gastroenterologist",
    "kidney": "Nephrologist", "urine": "Urologist",
    "mental": "Psychiatrist", "anxiety": "Psychiatrist", "depression": "Psychiatrist",
    "cancer": "Oncologist", "tumor": "Oncologist",
    "brain": "Neurologist", "headache": "Neurologist", "migraine": "Neurologist",
    "teeth": "Dentist", "dental": "Dentist",
    "pregnancy": "Gynecologist", "period": "Gynecologist",
}

def detect_specialty(question: str) -> str:
    q = question.lower()
    for keyword, specialty in SPECIALTY_MAP.items():
        if keyword in q:
            return specialty
    return "General Physician"

def get_practo_url(city: str, specialty: str) -> str:
    return f"https://www.practo.com/{city.strip().lower().replace(' ','-')}/{specialty.lower().replace(' ','-')}"

def get_justdial_url(city: str, specialty: str) -> str:
    return f"https://www.justdial.com/{city.strip().lower().replace(' ','+')}/{specialty.lower().replace(' ','+')}"

# ── Reverse geocode lat/lon → city using free API ─────────────────────────
def reverse_geocode(lat: float, lon: float) -> str:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "MediBot/1.0"},
            timeout=5
        )
        data = resp.json()
        addr = data.get("address", {})
        return addr.get("city") or addr.get("town") or addr.get("state_district") or "Unknown"
    except Exception:
        return ""

# ── GPS component ──────────────────────────────────────────────────────────
GPS_COMPONENT = """
<script>
navigator.geolocation.getCurrentPosition(
    function(pos) {
        const coords = pos.coords.latitude + "," + pos.coords.longitude;
        const el = window.parent.document.querySelector('[data-testid="stTextInput"] input[placeholder="GPS will fill this..."]');
        if(el){ el.value = coords; el.dispatchEvent(new Event('input', {bubbles:true})); }
    },
    function(err) { console.log("GPS error", err); }
);
</script>
"""

# ── Firebase REST helpers ──────────────────────────────────────────────────
FIREBASE_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"
FIRESTORE_BASE = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"

def firebase_signup(email: str, password: str) -> dict:
    r = requests.post(f"{FIREBASE_BASE}:signUp?key={FIREBASE_API_KEY}",
                      json={"email": email, "password": password, "returnSecureToken": True})
    return r.json()

def firebase_login(email: str, password: str) -> dict:
    r = requests.post(f"{FIREBASE_BASE}:signInWithPassword?key={FIREBASE_API_KEY}",
                      json={"email": email, "password": password, "returnSecureToken": True})
    return r.json()

def firestore_save(user_id: str, id_token: str, question: str, answer: str, city: str, specialty: str):
    now = datetime.datetime.utcnow().isoformat() + "Z"
    doc = {
        "fields": {
            "question":  {"stringValue": question},
            "answer":    {"stringValue": answer},
            "city":      {"stringValue": city},
            "specialty": {"stringValue": specialty},
            "timestamp": {"stringValue": now},
        }
    }
    requests.post(
        f"{FIRESTORE_BASE}/history_{user_id}",
        json=doc,
        headers={"Authorization": f"Bearer {id_token}"},
        params={"documentId": now.replace(":", "-")}
    )

def firestore_load(user_id: str, id_token: str) -> list:
    r = requests.get(
        f"{FIRESTORE_BASE}/history_{user_id}",
        headers={"Authorization": f"Bearer {id_token}"}
    )
    data = r.json()
    docs = data.get("documents", [])
    history = []
    for doc in docs:
        f = doc.get("fields", {})
        history.append({
            "question":  f.get("question",  {}).get("stringValue", ""),
            "answer":    f.get("answer",    {}).get("stringValue", ""),
            "city":      f.get("city",      {}).get("stringValue", ""),
            "specialty": f.get("specialty", {}).get("stringValue", ""),
            "time":      f.get("timestamp", {}).get("stringValue", ""),
        })
    return sorted(history, key=lambda x: x["time"])

# ── Session init ───────────────────────────────────────────────────────────
for key, val in {
    "logged_in": False, "user_id": "", "id_token": "",
    "user_email": "", "history": [], "city": "", "auth_mode": "Login"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load RAG pipeline ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MediBot engine…")
def load_pipeline():
    try:
        vs = load_vector_store()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    return initialize_rag_chain(vs)

qa_chain = load_pipeline()

# ══════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN (shown when not logged in)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div class="medibot-header">
        <h1>💊 MediBot</h1>
        <p>Your AI-powered Medical FAQ Assistant</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("🔐 Welcome to MediBot")

        mode = st.radio("", ["Login", "Sign Up"], horizontal=True, key="auth_mode_radio")
        email    = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")

        if mode == "Sign Up":
            confirm = st.text_input("🔒 Confirm Password", type="password")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not FIREBASE_API_KEY:
                    st.warning("Firebase not configured. See setup instructions below.")
                elif password != confirm:
                    st.error("Passwords don't match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    res = firebase_signup(email, password)
                    if "idToken" in res:
                        st.session_state.logged_in  = True
                        st.session_state.id_token   = res["idToken"]
                        st.session_state.user_id    = res["localId"]
                        st.session_state.user_email = email
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.error(res.get("error", {}).get("message", "Signup failed."))

        else:  # Login
            if st.button("Login", use_container_width=True, type="primary"):
                if not FIREBASE_API_KEY:
                    # Dev mode: skip auth
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email or "demo@medibot.com"
                    st.session_state.user_id    = "demo_user"
                    st.rerun()
                else:
                    res = firebase_login(email, password)
                    if "idToken" in res:
                        st.session_state.logged_in  = True
                        st.session_state.id_token   = res["idToken"]
                        st.session_state.user_id    = res["localId"]
                        st.session_state.user_email = email
                        st.session_state.history    = firestore_load(res["localId"], res["idToken"])
                        st.rerun()
                    else:
                        st.error(res.get("error", {}).get("message", "Login failed."))

        st.markdown("---")
        st.caption("🔒 Powered by Firebase Authentication")
        st.markdown('</div>', unsafe_allow_html=True)

    # Setup instructions if Firebase not configured
    if not FIREBASE_API_KEY:
        with st.expander("⚙️ Firebase Setup Instructions"):
            st.markdown("""
**Add these to your `.env` file (and Streamlit Cloud secrets):**
```
FIREBASE_API_KEY=your_web_api_key
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
```
**Steps:**
1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create project → Add Web App → copy config values above
3. Enable **Authentication → Email/Password**
4. Enable **Firestore Database** (Start in test mode)
""")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (shown after login)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.user_email}")
    st.caption("Logged in")

    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","user_id","id_token","user_email","history","city"]:
            st.session_state[k] = "" if k != "logged_in" else False
        st.session_state.history = []
        st.rerun()

    st.markdown("---")

    # ── Location ────────────────────────────────────────────────────────
    st.subheader("📍 Your Location")

    # GPS auto-detect button
    if st.button("📡 Detect My Location", use_container_width=True):
        st_html(GPS_COMPONENT, height=0)
        st.caption("Allow location access in your browser.")

    gps_raw = st.text_input("GPS will fill this...", key="gps_raw",
                             placeholder="GPS will fill this...", label_visibility="collapsed")

    if gps_raw and "," in gps_raw:
        try:
            lat, lon = map(float, gps_raw.split(","))
            detected = reverse_geocode(lat, lon)
            if detected:
                st.session_state.city = detected
                st.success(f"📌 Detected: **{detected}**")
        except Exception:
            pass

    manual_city = st.text_input("Or type your city:", placeholder="e.g. Mumbai, Delhi, Pune")
    if manual_city:
        st.session_state.city = manual_city
        st.success(f"📌 Set: **{manual_city}**")

    st.markdown("---")

    # ── History panel ────────────────────────────────────────────────────
    st.subheader("🕘 Past Questions")
    if st.session_state.history:
        for i, chat in enumerate(reversed(st.session_state.history[-15:])):
            label = (chat["question"][:35] + "…") if len(chat["question"]) > 35 else chat["question"]
            st.caption(f"🕐 {chat['time'][:16] if chat['time'] else ''}")
            st.markdown(f"**{label}**")
            st.caption(f"🏥 {chat.get('specialty','')}")
            st.markdown("---")
    else:
        st.caption("No history yet. Ask a question!")

    if st.session_state.history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.warning("⚠️ For informational purposes only.")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="medibot-header">
    <h1>🩺 MediBot – Health Assistant</h1>
    <p>Ask any medical question · Get answers · Find doctors near you</p>
</div>""", unsafe_allow_html=True)

# ── Chat history display ───────────────────────────────────────────────────
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["question"])
        st.caption(f"🕐 {chat.get('time', '')[:16]}")

    with st.chat_message("assistant", avatar="💊"):
        st.write(chat["answer"])
        if chat.get("city") and chat.get("specialty"):
            st.markdown(f'<div class="doctor-card">🏥 <b>Find a {chat["specialty"]}</b> in <b>{chat["city"]}</b></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔍 Practo", get_practo_url(chat["city"], chat["specialty"]), use_container_width=True)
            with c2:
                st.link_button("📋 JustDial", get_justdial_url(chat["city"], chat["specialty"]), use_container_width=True)

# ── Chat input ─────────────────────────────────────────────────────────────
user_question = st.chat_input("Type your medical question…")

if user_question and user_question.strip():
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    city      = st.session_state.city
    specialty = detect_specialty(user_question)

    with st.chat_message("user"):
        st.write(user_question)
        st.caption(f"🕐 {now}")

    with st.chat_message("assistant", avatar="💊"):
        with st.spinner("Thinking…"):
            answer, _ = ask_question_cached(qa_chain, user_question)
        st.write(answer)

        if city:
            st.markdown(f'<div class="doctor-card">🏥 <b>Find a {specialty}</b> in <b>{city}</b></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔍 Practo", get_practo_url(city, specialty), use_container_width=True)
            with c2:
                st.link_button("📋 JustDial", get_justdial_url(city, specialty), use_container_width=True)
        else:
            st.info("📍 Enter your city in the sidebar to find doctors near you.")

    entry = {"question": user_question, "answer": answer,
             "city": city, "specialty": specialty, "time": now}
    st.session_state.history.append(entry)

    # Save to Firestore if logged in with Firebase
    if st.session_state.id_token and FIREBASE_API_KEY:
        firestore_save(st.session_state.user_id, st.session_state.id_token,
                       user_question, answer, city, specialty)