import asyncio
from gtts import gTTS
import io
from fastapi.responses import StreamingResponse
from fastapi import FastAPI,HTTPException,APIRouter

generate_audio_router = APIRouter()

valid_languages = {
    "punjabi":"pa",
    "english":"en",
    "hindi":"hi"
}

@generate_audio_router.get("/generate_audio")
async def generate_audio_endpoint(answer: str, language: str):
    """Endpoint to query the RAG store and get an answer."""
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")
    if language not in valid_languages:
        raise HTTPException(status_code=400, detail=f"Invalid language. Supported languages: {', '.join(valid_languages.keys())}")
    try:
        tts = gTTS(text=answer, lang=valid_languages[language])
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    return StreamingResponse(
        content=buffer,
        media_type="audio/mpeg",
        headers={"language":language,"Content-Disposition": "inline; filename=answer.mp3"}
    )