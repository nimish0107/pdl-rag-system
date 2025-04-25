import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from utils import logger  


from RAG.TextSplitter import MultilingualTextSplitter

LANGUAGES = ["punjabi", "hindi", "english"]
MODEL_NAME = "intfloat/multilingual-e5-small"  
EMBEDDING_DIR = "faiss_indexes"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

class MultilingualEmbedder:

    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts using the SentenceTransformer model.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Check if embeddings are empty or None
        if embeddings is None or len(embeddings) == 0:
            logger.warning(" No embeddings returned.")
            return None
        
        logger.info(f"Generated {len(embeddings)} embeddings with shape: {embeddings.shape}")
        return embeddings

class FaissEmbeddingStore:
    def __init__(self, model_name=MODEL_NAME, persist_dir=EMBEDDING_DIR):
        """
        Initializes the FaissEmbeddingStore with a multilingual embedder and FAISS indexes for each language.
        """
        self.persist_dir = persist_dir
        self.embedder = MultilingualEmbedder(model_name)
        self.indexes = {lang: faiss.IndexFlatL2(384) for lang in LANGUAGES}  
        self.metadata = {lang: [] for lang in LANGUAGES}
    
    def add_documents(self, chunked_docs: Dict[str, List[Dict]]):
        """
        Embeds documents for each language and stores embeddings in FAISS index.
        """
        for lang in LANGUAGES:
            # Extract text for the current language
            texts = [doc["text"] for doc in chunked_docs.get(lang, [])]
            
            if not texts:
                logger.warning(f" No text to embed for {lang}")
                continue
            
            logger.info(f"Generating embeddings for {lang}...")
            
            # Generate embeddings using the SentenceTransformer model
            embeddings = self.embedder.embed(texts)
            
            # Check if embeddings are empty or None
            if embeddings is None or len(embeddings) == 0:
                logger.warning(f" No embeddings returned for {lang}.")
                continue

            logger.info(f"Generated {len(embeddings)} embeddings for {lang} with shape: {embeddings.shape}")
            
            # Add embeddings to the FAISS index for the current language
            self.indexes[lang].add(embeddings)

            # Store metadata
            for i, doc in enumerate(chunked_docs.get(lang, [])):
                self.metadata[lang].append({
                    "chunk_id": doc["chunk_id"],
                    "doc_id": doc["doc_id"],
                    "chunk_idx": doc["chunk_idx"],
                    "text": doc["text"],
                    "parallel_id": doc.get("parallel_id", None)
                })

            # Log the chunks for validation purposes
            logger.info(f"Chunks for {lang}:")
            for chunk in chunked_docs.get(lang, []):
                logger.info(f"Chunk ID: {chunk['chunk_id']}, Doc ID: {chunk['doc_id']}, Text: {chunk['text'][:50]}...")

    def save_all(self):
        """
        Saves the FAISS index and metadata for each language.
        """
        for lang in LANGUAGES:
            # Save the FAISS index to a file
            faiss.write_index(self.indexes[lang], os.path.join(EMBEDDING_DIR, f"{lang}_index.faiss"))
            
            # Save metadata to a JSON file
            with open(os.path.join(EMBEDDING_DIR, f"{lang}_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(self.metadata[lang], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {lang} index and metadata to {EMBEDDING_DIR}")
            

