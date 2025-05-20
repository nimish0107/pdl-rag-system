import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException,APIRouter
from services import add_document
import uuid

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

add_document_router = APIRouter()

@add_document_router.post("/add_document")
async def add_document_endpoint(file: UploadFile = File(...)):
    # Validate uploaded file
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    doc_uuid = str(uuid.uuid4())
    # Save file to data directory
    file_path = os.path.join(DATA_DIR, doc_uuid + "_" + file.filename)
    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File already exists.")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = await add_document(file_path, doc_uuid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")

    return result