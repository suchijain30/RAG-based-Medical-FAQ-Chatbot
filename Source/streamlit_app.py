"""
streamlit_app.py – MediBot web interface with doctor recommendations.
Run: streamlit run src/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_pipeline import load_vector_store, initialize_rag_chain, ask_question_cached

st.set_page_config(
    page_title="MediBot – Medical FAQ Chatbot",
    page_icon="💊",
    layout="centered"
)

# ── Doctor specialty mapping based on keywords ────────────────────────────────
SPECIALTY_MAP = {
    "heart": "Cardiologist",
    "chest pain": "Cardiologist",
    "blood pressure": "Cardiologist",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "sugar": "Endocrinologist",
    "skin": "Dermatologist",
    "rash": "Dermatologist",
    "acne": "Dermatologist",
    "bone": "Orthopedic",
    "joint": "Orthopedic",
    "knee": "Orthopedic",
    "back pain": "Orthopedic",
    "eye": "Ophthalmologist",
    "vision": "Ophthalmologist",
    "ear": "ENT Specialist",
    "throat": "ENT Specialist",
    "nose": "ENT Specialist",
    "child": "Pediatrician",
    "fever": "General Physician",
    "cold": "General Physician",
    "cough": "Pulmonologist",
    "lung": "Pulmonologist",
    "breathing": "Pulmonologist",
    "stomach": "Gastroenterologist",
    "liver": "Gastroenterologist",
    "digestion": "Gastroenterologist",
    "kidney": "Nephrologist",
    "urine": "Urologist",
    "mental": "Psychiatrist",
    "anxiety": "Psychiatrist",
    "depression": "Psychiatrist",
    "cancer": "Oncologist",
    "tumor": "Oncologist",
    "brain": "Neurologist",
    "headache": "Neurologist",
    "migraine": "Neurologist",
    "teeth": "Dentist",
    "dental": "Dentist",
    "pregnancy": "Gynecologist",
    "period": "Gynecologist",
    "women": "Gynecologist",
}

def detect_specialty(question: str) -> str:
    q = question.lower()
    for keyword, specialty in SPECIALTY_MAP.items():
        if keyword in q:
            return specialty
    return "General Physician"


def get_practo_url(city: str, specialty: str) -> str:
    city_slug = city.strip().lower().replace(" ", "-")
    specialty_slug = specialty.lower().replace(" ", "-")
    return f"https://www.practo.com/{city_slug}/{specialty_slug}"


def get_justdial_url(city: str, specialty: str) -> str:
    city_slug = city.strip().lower().replace(" ", "+")
    specialty_slug = specialty.lower().replace(" ", "+")
    return f"https://www.justdial.com/{city_slug}/{specialty_slug}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💊 MediBot")
    st.caption("Powered by RAG · FAISS · Groq LLaMA")
    st.markdown("---")

    st.subheader("📍 Your Location")
    user_city = st.text_input(
        "Enter your city for doctor recommendations:",
        placeholder="e.g. Mumbai, Delhi, Pune",
        key="city_input"
    )
    if user_city:
        st.success(f"Location set: **{user_city}**")

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** For informational purposes only. "
        "Always consult a qualified healthcare professional."
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()


# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing MediBot…")
def load_pipeline():
    try:
        vectorstore = load_vector_store()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    return initialize_rag_chain(vectorstore)

qa_chain = load_pipeline()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 MediBot – Health Assistant")
st.caption("Ask any medical question. Get answers + find doctors near you.")
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Chat history ──────────────────────────────────────────────────────────────
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant", avatar="💊"):
        st.write(chat["answer"])

        # Doctor recommendation card
        if chat.get("city") and chat.get("specialty"):
            city = chat["city"]
            specialty = chat["specialty"]
            st.markdown("---")
            st.markdown(f"#### 🏥 Find a **{specialty}** near you in **{city}**")
            col1, col2 = st.columns(2)
            with col1:
                practo_url = get_practo_url(city, specialty)
                st.link_button("🔍 Search on Practo", practo_url, use_container_width=True)
            with col2:
                jd_url = get_justdial_url(city, specialty)
                st.link_button("📋 Search on JustDial", jd_url, use_container_width=True)

        with st.expander("📄 Retrieved Sources"):
            for doc in chat.get("sources", []):
                q = doc.metadata.get("question", "N/A")
                src = doc.metadata.get("source", "N/A")
                st.markdown(f"- **Q:** {q}  `[{src}]`")


# ── Input ──────────────────────────────────────────────────────────────────────
user_question = st.chat_input("Type your medical question…")

if user_question and user_question.strip():
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant", avatar="💊"):
        with st.spinner("Thinking…"):
            answer, sources = ask_question_cached(qa_chain, user_question)

        st.write(answer)

        # Detect specialty and show doctor links
        specialty = detect_specialty(user_question)
        city = st.session_state.get("city_input", "").strip()

        if city:
            st.markdown("---")
            st.markdown(f"#### 🏥 Find a **{specialty}** near you in **{city}**")
            col1, col2 = st.columns(2)
            with col1:
                practo_url = get_practo_url(city, specialty)
                st.link_button("🔍 Search on Practo", practo_url, use_container_width=True)
            with col2:
                jd_url = get_justdial_url(city, specialty)
                st.link_button("📋 Search on JustDial", jd_url, use_container_width=True)
        else:
            st.info("📍 Enter your city in the sidebar to find doctors near you.")

        with st.expander("📄 Retrieved Sources"):
            for doc in sources:
                q = doc.metadata.get("question", "N/A")
                src = doc.metadata.get("source", "N/A")
                st.markdown(f"- **Q:** {q}  `[{src}]`")

    st.session_state.history.append({
        "question": user_question,
        "answer": answer,
        "sources": sources,
        "city": city,
        "specialty": specialty
    })