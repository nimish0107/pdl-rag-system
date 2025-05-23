import os
import shutil
import asyncio
from fastapi import FastAPI,HTTPException,APIRouter
from services import query_chatbot

query_chatbot_router = APIRouter()

@query_chatbot_router.get("/query")
async def query_endpoint(query: str, language: str):
    """Endpoint to query the RAG store and get an answer."""
    try:
        answer = await query_chatbot(query, language)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    return {"status": "success", "query": query, "language": language, "answer": answer}


#     buffer = io.BytesIO()
#     sf.write(buffer, audio, 16000, format='WAV')  # Adjust sample rate if needed
#     buffer.seek(0)

#     return StreamingResponse(
#         content=buffer,
#         media_type="audio/wav",
#         headers={"answer": answer, "query":query,"language":language,"Content-Disposition": "inline; filename=answer.wav"}
#     )