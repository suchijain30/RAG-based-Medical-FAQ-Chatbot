"""
streamlit_app.py - MediBot | No login | Auto GPS | Doctor links only when city known
Run: streamlit run src/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import datetime
import requests
from streamlit.components.v1 import html as st_html
from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached

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
    "stomach": "Gastroenterologist", "liver": "Gastroenterologist", "digestion": "Gastroenterologist",
    "kidney": "Nephrologist", "urine": "Urologist",
    "mental": "Psychiatrist", "anxiety": "Psychiatrist", "depression": "Psychiatrist",
    "cancer": "Oncologist", "tumor": "Oncologist",
    "brain": "Neurologist", "headache": "Neurologist", "migraine": "Neurologist",
    "teeth": "Dentist", "dental": "Dentist",
    "pregnancy": "Gynecologist", "period": "Gynecologist",
    "dengue": "General Physician", "malaria": "General Physician", "typhoid": "General Physician",
}

def detect_specialty(question: str) -> str:
    q = question.lower()
    for keyword, specialty in SPECIALTY_MAP.items():
        if keyword in q:
            return specialty
    return "General Physician"

def get_practo_url(city: str, specialty: str) -> str:
    return f"https://www.practo.com/{city.strip().lower().replace(' ', '-')}/{specialty.lower().replace(' ', '-')}"

def get_justdial_url(city: str, specialty: str) -> str:
    return f"https://www.justdial.com/{city.strip().lower().replace(' ', '+')}/{specialty.lower().replace(' ', '+')}"

def reverse_geocode(lat: float, lon: float) -> str:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "MediBot/1.0"},
            timeout=5
        )
        addr = r.json().get("address", {})
        # Priority: city → town → district → state
        return (addr.get("city") or addr.get("town") or
                addr.get("state_district") or addr.get("state") or "")
    except Exception:
        return ""

# ── Session init ───────────────────────────────────────────────────────────
for key, val in {
    "history": [], "city": "", "gps_done": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Auto GPS on page load (runs once) ─────────────────────────────────────
# Injects JS that posts coords into a hidden Streamlit text input
GPS_JS = """
<script>
(function() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(pos) {
            var lat = pos.coords.latitude.toFixed(5);
            var lon = pos.coords.longitude.toFixed(5);
            // Write to the hidden input Streamlit renders for gps_coords key
            var inputs = window.parent.document.querySelectorAll('input[type="text"]');
            inputs.forEach(function(inp) {
                if (inp.placeholder === '__gps__') {
                    inp.value = lat + ',' + lon;
                    inp.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });
        }, function(err) {
            console.log('GPS denied or unavailable:', err.message);
        }, { timeout: 8000 });
    }
})();
</script>
"""

# Render GPS JS once on load
if not st.session_state.gps_done:
    st_html(GPS_JS, height=0)

# Hidden input to receive GPS coords from JS
gps_raw = st.text_input("gps", placeholder="__gps__",
                         key="gps_coords", label_visibility="collapsed")

# Process coords the moment they arrive
if gps_raw and "," in gps_raw and not st.session_state.gps_done:
    try:
        lat, lon = map(float, gps_raw.split(","))
        detected_city = reverse_geocode(lat, lon)
        if detected_city:
            st.session_state.city = detected_city
        st.session_state.gps_done = True
    except Exception:
        st.session_state.gps_done = True

# ── Load pipeline ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MediBot…")
def load_pipeline():
    try:
        vs = load_vector_store()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    return initialize_rag_chain(vs)

qa_chain = load_pipeline()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 MediBot")
    st.caption("RAG · FAISS · Groq LLaMA-3.3")
    st.markdown("---")

    st.markdown("### 📍 Location")
    if st.session_state.city:
        st.success(f"📌 **{st.session_state.city}**")
        st.caption("Auto-detected via GPS")
    else:
        st.warning("📡 Detecting location…\n\nAllow browser location access.")

    # Always allow manual override
    manual = st.text_input("Override city:", placeholder="e.g. Mumbai, Delhi")
    if manual.strip():
        st.session_state.city = manual.strip()

    st.markdown("---")
    st.markdown("### 🕘 History")
    if st.session_state.history:
        for chat in reversed(st.session_state.history[-10:]):
            label = chat["question"][:38] + "…" if len(chat["question"]) > 38 else chat["question"]
            st.caption(f"🕐 {chat['time']}")
            st.markdown(f"**{label}**")
            st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No history yet.")

    st.markdown("---")
    st.warning("⚠️ For informational purposes only. Consult a doctor for personal advice.")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="medibot-header">
    <h1>🩺 MediBot – AI Health Assistant</h1>
    <p>Ask any medical question · Powered by Agentic RAG · Groq LLaMA-3.3</p>
</div>""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["question"])
        st.caption(f"🕐 {chat['time']}")

    with st.chat_message("assistant", avatar="💊"):
        st.write(chat["answer"])

        # Show doctor links only if city was known at time of question
        if chat.get("city") and chat.get("specialty"):
            st.markdown(
                f'<div class="doctor-card">🏥 <b>Find a {chat["specialty"]}</b>'
                f' near you in <b>{chat["city"]}</b></div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔍 Practo",
                    get_practo_url(chat["city"], chat["specialty"]),
                    use_container_width=True)
            with c2:
                st.link_button("📋 JustDial",
                    get_justdial_url(chat["city"], chat["specialty"]),
                    use_container_width=True)

# ── Chat input ─────────────────────────────────────────────────────────────
user_question = st.chat_input("Type your medical question…")

if user_question and user_question.strip():
    now       = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    city      = st.session_state.city
    specialty = detect_specialty(user_question)

    with st.chat_message("user"):
        st.write(user_question)
        st.caption(f"🕐 {now}")

    with st.chat_message("assistant", avatar="💊"):
        with st.spinner("Thinking…"):
            answer, _ = ask_question_cached(qa_chain, user_question)
        st.write(answer)

        # Doctor links — only if city is available
        if city:
            st.markdown(
                f'<div class="doctor-card">🏥 <b>Find a {specialty}</b>'
                f' near you in <b>{city}</b></div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔍 Practo",
                    get_practo_url(city, specialty),
                    use_container_width=True)
            with c2:
                st.link_button("📋 JustDial",
                    get_justdial_url(city, specialty),
                    use_container_width=True)
        # If no city — show nothing (no info banner either, keep it clean)

    st.session_state.history.append({
        "question": user_question,
        "answer":   answer,
        "city":     city,
        "specialty": specialty,
        "time":     now,
    })