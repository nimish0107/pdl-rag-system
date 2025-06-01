import asyncio
from OCR.ocr import read_image, image_to_text
from Translation.translate import translate_punjabi_to_HindiEnglish
from RAG.TextSplitter import MultilingualTextSplitter, CHUNK_SIZE, CHUNK_OVERLAP
from RAG.embeddings import FaissEmbeddingStore
from RAG.generation import OllamaAnswerGenerator
from typing import List, Dict, AsyncGenerator
# from TTS.tts_engine import synthesize_speech
from utils import logger

store = FaissEmbeddingStore()
text_splitter = MultilingualTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

llm_model_name = "pdlRAG"
llm_base_url = "http://localhost:11434"
answer_generator = OllamaAnswerGenerator(model_name=llm_model_name, ollama_base_url=llm_base_url)
async def add_document(file_path: str, doc_uuid: str) -> int:
    """
    Pipeline to process an image document:
    1. Perform OCR to extract text
    2. Translate extracted text to Hindi and English
    3. Split multilingual text into chunks
    4. Generate embeddings and add to Faiss store
    """
    # 1. Read and OCR (synchronous)
    logger.info("Reading Images")
    image = await asyncio.to_thread(read_image, file_path)
    raw_text = await asyncio.to_thread(image_to_text, image)

    # 2. Translate (async)
    logger.info("Translating text")
    translations = await translate_punjabi_to_HindiEnglish(raw_text)
    logger.info(translations)
    # Prepare a unified document structure
    doc = {
        "punjabi": [raw_text],
        "hindi": translations.get("hindi", ""),
        "english": translations.get("english", "")
    }

    # 3. Split into chunks (sync) -> wrap in thread
    logger.info("Splitting text into chunks")
    chunked_docs = await asyncio.to_thread(text_splitter.split_documents, doc, doc_uuid)

    # 4. Generate embeddings and store (sync) -> wrap in thread
    logger.info("Adding documents to store")
    await asyncio.to_thread(store.add_documents, chunked_docs)
    
    return len(chunked_docs['punjabi'])

async def query_chatbot(query: str, language: str, k: int = 6) -> AsyncGenerator[str, None]:
    """
    Retrieve relevant chunks from store and generate an answer.
    """
    # Validate language
    if language not in {"punjabi", "hindi", "english"}:
        raise ValueError(f"Unsupported language '{language}'. Valid options are punjabi, hindi, english.")

    # Search the FAISS store (sync -> thread)
    results = await asyncio.to_thread(store.search, query, language, k)

    # Generate answer from RAG model (sync -> thread)
    async for chunk in answer_generator.generate_answer(query, results, language):
        yield chunk

# async def generate_audio(text: str, language: str) -> str:
#     """
#     Retrieve relevant chunks from store and generate an answer.
#     """
#     # Validate language
#     if language not in {"punjabi", "hindi", "english"}:
#         raise ValueError(f"Unsupported language '{language}'. Valid options are punjabi, hindi, english.")

#     audio, sr = await asyncio.to_thread(synthesize_speech, text, language)
#     return audio, sr

async def delete_doc_by_id(doc_id: str) -> bool:
    """
    Delete a document by its ID.
    """
    # Validate document ID
    if not doc_id:
        raise ValueError("Invalid document ID.")

    # Delete the document (sync -> thread)
    result = await asyncio.to_thread(store.delete_document_by_id, doc_id)
    return result

async def delete_all_docs() -> bool:
    """
    Asynchronously delete all documents from all language FAISS indexes.
    """
    result = await asyncio.to_thread(store.delete_all_documents)
    return result