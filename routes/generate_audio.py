import os
import shutil
import asyncio
import soundfile as sf
import io
from fastapi.responses import StreamingResponse
from fastapi import FastAPI,HTTPException,APIRouter
from services import generate_audio

generate_audio_router = APIRouter()

@generate_audio_router.get("/generate_audio")
async def generate_audio_endpoint(answer: str, language: str):
    """Endpoint to query the RAG store and get an answer."""
    try:
        audio = await generate_audio(answer, language)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format='WAV')  # Adjust sample rate if needed
    buffer.seek(0)

    return StreamingResponse(
        content=buffer,
        media_type="audio/wav",
        headers={"language":language,"Content-Disposition": "inline; filename=answer.wav"}
    )