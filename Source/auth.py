"""
auth.py - Firebase Authentication + Firestore history manager
"""

import os
import requests
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

FIREBASE_API_KEY    = os.getenv("FIREBASE_API_KEY", "")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

FIREBASE_AUTH_BASE  = "https://identitytoolkit.googleapis.com/v1/accounts"
FIRESTORE_BASE      = (
    f"https://firestore.googleapis.com/v1/projects/"
    f"{FIREBASE_PROJECT_ID}/databases/(default)/documents"
)


class FirebaseAuthUnavailable(Exception):
    """Raised when Firebase cannot validate an authentication token."""


class FirestoreRequestError(Exception):
    """Raised when Firestore rejects or cannot complete a data request."""


def _check_firestore_response(response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        if not detail:
            detail = response.text[:300] or "No error details returned."
        raise FirestoreRequestError(
            f"Firestore {action} failed ({response.status_code}): {detail}"
        ) from exc

# ── Auth ───────────────────────────────────────────────────────────────────

def verify_id_token(id_token: str) -> str | None:
    """Return the Firebase user ID for a valid ID token, or None if invalid."""
    if not FIREBASE_API_KEY:
        raise FirebaseAuthUnavailable("Firebase API key is not configured.")

    try:
        response = requests.post(
            f"{FIREBASE_AUTH_BASE}:lookup?key={FIREBASE_API_KEY}",
            json={"idToken": id_token},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise FirebaseAuthUnavailable("Unable to reach Firebase Authentication.") from exc

    if response.status_code != 200:
        return None

    try:
        users = response.json().get("users", [])
    except requests.exceptions.JSONDecodeError as exc:
        raise FirebaseAuthUnavailable("Firebase returned an invalid authentication response.") from exc
    return users[0].get("localId") if users else None


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


def _firestore_headers(id_token: str) -> dict:
    return {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json",
    }


def load_profile(user_id: str, id_token: str) -> dict:
    response = requests.get(
        f"{FIRESTORE_BASE}/users/{user_id}",
        headers=_firestore_headers(id_token),
        timeout=10,
    )
    if response.status_code == 404:
        return {}
    _check_firestore_response(response, "profile load")
    fields = response.json().get("fields", {})
    return {key: value.get("stringValue", "") for key, value in fields.items()}


def save_profile(user_id: str, id_token: str, profile: dict) -> dict:
    response = requests.patch(
        f"{FIRESTORE_BASE}/users/{user_id}",
        headers=_firestore_headers(id_token),
        json={"fields": {
            key: {"stringValue": str(value)}
            for key, value in profile.items()
        }},
        timeout=10,
    )
    _check_firestore_response(response, "profile save")
    return profile


def save_message(user_id: str, id_token: str,
                 role: str, content: str,
                 city: str = "", specialty: str = "") -> bool:
    """Save a single message (user or assistant) to Firestore."""
    now = datetime.utcnow().isoformat() + "Z"
    doc_id = uuid4().hex

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
        headers=_firestore_headers(id_token),
        params={"documentId": doc_id},
        timeout=10
    )
    r.raise_for_status()
    return True


def load_recent_messages(user_id: str, id_token: str, limit: int = 100) -> list[dict]:
    response = requests.get(
        f"{FIRESTORE_BASE}/users/{user_id}/messages",
        headers=_firestore_headers(id_token),
        params={"pageSize": limit, "orderBy": "timestamp desc"},
        timeout=15,
    )
    response.raise_for_status()
    messages = []
    for doc in response.json().get("documents", []):
        fields = doc.get("fields", {})
        messages.append({
            key: fields.get(key, {}).get("stringValue", "")
            for key in ("role", "content", "city", "specialty", "timestamp")
        })
    messages.reverse()
    return messages


def load_all_messages(user_id: str, id_token: str) -> list[dict]:
    """Load ALL messages for a user, sorted oldest first."""
    messages = []
    url = f"{FIRESTORE_BASE}/users/{user_id}/messages"
    params = {"pageSize": 1000}
    while url:
        response = requests.get(url, headers=_firestore_headers(id_token), params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        for doc in data.get("documents", []):
            fields = doc.get("fields", {})
            messages.append({
                key: fields.get(key, {}).get("stringValue", "")
                for key in ("role", "content", "city", "specialty", "timestamp")
            })
        token = data.get("nextPageToken")
        url = f"{FIRESTORE_BASE}/users/{user_id}/messages" if token else ""
        params = {"pageSize": 1000, "pageToken": token} if token else {}

    messages.sort(key=lambda x: x["timestamp"])
    return messages


def delete_history(user_id: str, id_token: str) -> bool:
    """Delete all messages for a user (used by 'Clear History' button)."""
    url = f"{FIRESTORE_BASE}/users/{user_id}/messages"
    params = {"pageSize": 1000}
    while url:
        response = requests.get(url, headers=_firestore_headers(id_token), params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        for doc in data.get("documents", []):
            name = doc.get("name", "")
            deleted = requests.delete(
                f"https://firestore.googleapis.com/v1/{name}",
                headers=_firestore_headers(id_token),
                timeout=10,
            )
            deleted.raise_for_status()
        token = data.get("nextPageToken")
        url = f"{FIRESTORE_BASE}/users/{user_id}/messages" if token else ""
        params = {"pageSize": 1000, "pageToken": token} if token else {}
    return True