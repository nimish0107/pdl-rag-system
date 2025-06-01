import os
import shutil
import asyncio
from fastapi import FastAPI,HTTPException,APIRouter
from services import query_chatbot
from fastapi.responses import StreamingResponse

query_chatbot_router = APIRouter()

@query_chatbot_router.get("/query")
async def query_endpoint(query: str, language: str):
    """Endpoint to query the RAG store and get an answer."""
    try:
        # answer_generator = query_chatbot(query, language)

        # async def answer_streamer():
        #     async for chunk in answer_generator:
        #         # If chunk is string, encode it
        #         yield chunk.encode("utf-8")

        return StreamingResponse(
            query_chatbot(query, language),
            media_type="text/plain",  # or "application/json" if you want JSON chunks
            headers={"Cache-Control": "no-cache"},
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    # return {"status": "success", "query": query, "language": language, "answer": answer}


#     buffer = io.BytesIO()
#     sf.write(buffer, audio, 16000, format='WAV')  # Adjust sample rate if needed
#     buffer.seek(0)

#     return StreamingResponse(
#         content=buffer,
#         media_type="audio/wav",
#         headers={"answer": answer, "query":query,"language":language,"Content-Disposition": "inline; filename=answer.wav"}
#     )