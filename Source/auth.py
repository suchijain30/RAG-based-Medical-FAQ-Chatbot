"""
auth.py - Firebase Authentication + Firestore history manager
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

FIREBASE_API_KEY    = os.getenv("FIREBASE_API_KEY", "")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

FIREBASE_AUTH_BASE  = "https://identitytoolkit.googleapis.com/v1/accounts"
FIRESTORE_BASE      = (
    f"https://firestore.googleapis.com/v1/projects/"
    f"{FIREBASE_PROJECT_ID}/databases/(default)/documents"
)


# ── Auth ───────────────────────────────────────────────────────────────────

def signup(email: str, password: str) -> dict:
    """Create new account. Returns {ok, id_token, user_id, error}"""
    r = requests.post(
        f"{FIREBASE_AUTH_BASE}:signUp?key={FIREBASE_API_KEY}",
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=10
    )
    data = r.json()
    if "idToken" in data:
        return {"ok": True, "id_token": data["idToken"],
                "user_id": data["localId"], "email": email}
    msg = data.get("error", {}).get("message", "Signup failed.")
    return {"ok": False, "error": _friendly_error(msg)}


def login(email: str, password: str) -> dict:
    """Sign in existing account. Returns {ok, id_token, user_id, error}"""
    r = requests.post(
        f"{FIREBASE_AUTH_BASE}:signInWithPassword?key={FIREBASE_API_KEY}",
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=10
    )
    data = r.json()
    if "idToken" in data:
        return {"ok": True, "id_token": data["idToken"],
                "user_id": data["localId"], "email": email}
    msg = data.get("error", {}).get("message", "Login failed.")
    return {"ok": False, "error": _friendly_error(msg)}


def _friendly_error(msg: str) -> str:
    mapping = {
        "EMAIL_EXISTS":           "This email is already registered. Please log in.",
        "EMAIL_NOT_FOUND":        "No account found with this email.",
        "INVALID_PASSWORD":       "Incorrect password. Please try again.",
        "INVALID_EMAIL":          "Invalid email address.",
        "WEAK_PASSWORD":          "Password must be at least 6 characters.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
        "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
    }
    for key, friendly in mapping.items():
        if key in msg:
            return friendly
    return msg


# ── Firestore history ──────────────────────────────────────────────────────

def _headers(id_token: str) -> dict:
    return {"Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"}


def save_message(user_id: str, id_token: str,
                 role: str, content: str,
                 city: str = "", specialty: str = "") -> bool:
    """Save a single message (user or assistant) to Firestore."""
    now = datetime.utcnow().isoformat() + "Z"
    doc_id = now.replace(":", "-").replace(".", "-")  # valid Firestore doc ID

    doc = {
        "fields": {
            "role":      {"stringValue": role},
            "content":   {"stringValue": content},
            "city":      {"stringValue": city},
            "specialty": {"stringValue": specialty},
            "timestamp": {"stringValue": now},
        }
    }

    r = requests.post(
        f"{FIRESTORE_BASE}/users/{user_id}/messages",
        json=doc,
        headers=_headers(id_token),
        params={"documentId": doc_id},
        timeout=10
    )
    return r.status_code in (200, 201)


def load_all_messages(user_id: str, id_token: str) -> list[dict]:
    """Load ALL messages for a user, sorted oldest first."""
    r = requests.get(
        f"{FIRESTORE_BASE}/users/{user_id}/messages",
        headers=_headers(id_token),
        params={"pageSize": 1000,
                "orderBy": "timestamp"},   # needs Firestore index (auto-created)
        timeout=15
    )
    data = r.json()
    docs = data.get("documents", [])

    messages = []
    for doc in docs:
        f = doc.get("fields", {})
        messages.append({
            "role":      f.get("role",      {}).get("stringValue", ""),
            "content":   f.get("content",   {}).get("stringValue", ""),
            "city":      f.get("city",      {}).get("stringValue", ""),
            "specialty": f.get("specialty", {}).get("stringValue", ""),
            "timestamp": f.get("timestamp", {}).get("stringValue", ""),
        })

    # Sort by timestamp (Firestore orderBy needs index; fallback sort here)
    messages.sort(key=lambda x: x["timestamp"])
    return messages


def delete_history(user_id: str, id_token: str) -> bool:
    """Delete all messages for a user (used by 'Clear History' button)."""
    # Firestore doesn't support bulk delete via REST easily —
    # we fetch all doc names and delete one by one
    r = requests.get(
        f"{FIRESTORE_BASE}/users/{user_id}/messages",
        headers=_headers(id_token),
        params={"pageSize": 1000},
        timeout=15
    )
    docs = r.json().get("documents", [])
    for doc in docs:
        doc_name = doc.get("name", "")
        requests.delete(
            f"https://firestore.googleapis.com/v1/{doc_name}",
            headers=_headers(id_token),
            timeout=10
        )
    return True