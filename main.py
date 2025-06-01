from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.add_document import add_document_router
from routes.query_chatbot import query_chatbot_router
from routes.generate_audio import generate_audio_router
from routes.delete_doc_by_id import delete_document_router
from routes.delete_all_docs import delete_all_docs_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
app.include_router(add_document_router)
app.include_router(query_chatbot_router)
app.include_router(generate_audio_router)
app.include_router(delete_document_router)
app.include_router(delete_all_docs_router)