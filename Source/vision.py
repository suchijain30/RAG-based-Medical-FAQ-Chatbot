"""
vision.py - Phase 3B: Medical image analysis using Groq Vision API
Supports: skin rashes, prescriptions, lab reports, symptom photos
"""

import os
import base64
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Use the same model already in use — qwen3.6-27b supports vision on Groq
VISION_MODEL = "qwen/qwen3.6-27b"


def analyze_medical_image(
    image_bytes: bytes,
    user_question: str = "",
    mime_type: str = "image/jpeg",
) -> str:
    """
    Analyze a medical image using Groq Vision API.

    Args:
        image_bytes: Raw bytes of the image file
        user_question: Optional user question about the image
        mime_type: MIME type of the image (image/jpeg, image/png, etc.)

    Returns:
        Structured medical analysis text or error message
    """
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set. Cannot analyze images."

    if not image_bytes:
        return "⚠️ No image data received. Please upload a valid image."

    # Encode image to base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_image}"

    # Build the analysis prompt
    system_prompt = (
        "You are a medical image analysis assistant. "
        "Analyze the uploaded image and provide:\n\n"
        "1. **What I see**: Describe the visible features (skin condition, "
        "text on prescription, lab values, etc.)\n"
        "2. **Possible observations**: Based on visual features, what conditions "
        "or findings might be relevant (if applicable)\n"
        "3. **Severity cues**: Any visible indicators of severity (mild, moderate, severe)\n"
        "4. **Recommended action**: What type of doctor to see, or next steps\n\n"
        "IMPORTANT RULES:\n"
        "- Never provide a definitive diagnosis — only observations and possibilities\n"
        "- Always end with: 'Please consult a healthcare professional for proper diagnosis.'\n"
        "- If the image is not medical (e.g., a landscape, food), say so politely\n"
        "- If the image is unclear or low quality, say so and ask for a better image\n"
        "- For prescriptions: read and explain the medications, dosages, and purposes\n"
        "- For lab reports: highlight any values outside normal range\n"
    )

    user_text = user_question.strip() if user_question else "Please analyze this medical image."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content.strip()
        if not answer:
            return "⚠️ Could not analyze the image. Please try a clearer photo."
        return answer

    except RateLimitError:
        return (
            "⚠️ Groq rate limit reached. Please wait ~60 seconds and try again."
        )
    except Exception as e:
        err = str(e)
        if "image" in err.lower() or "base64" in err.lower():
            return "⚠️ Image format not supported. Please use JPG, PNG, or WebP."
        return f"⚠️ Image analysis error: {err}"


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")
