from langchain_community.vectorstores import FAISS
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from utils import logger  
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


from RAG.TextSplitter import MultilingualTextSplitter

LANGUAGES = ["punjabi", "hindi", "english"]
MODEL_NAME = "intfloat/multilingual-e5-small"  
EMBEDDING_DIR = "faiss_indexes"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

class MultilingualEmbedder(Embeddings):

    def __init__(self, model_name=MODEL_NAME):
        """
        Initializes the SentenceTransformer model for multilingual embeddings.
        """
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of Document objects using the SentenceTransformer model.
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query using the SentenceTransformer model.
        """
        return self.model.encode([query], convert_to_numpy=True, show_progress_bar=True)[0]

class FaissEmbeddingStore:
    def __init__(self, model_name=MODEL_NAME, persist_dir=EMBEDDING_DIR):
        """
        Initializes the FaissEmbeddingStore with a multilingual embedder and FAISS indexes for each language.
        """
        self.persist_dir = persist_dir
        self.embedder = MultilingualEmbedder(model_name)
        self.vector_stores = {}

        for lang in LANGUAGES:
            self.vector_stores[lang] = None
            self._load_vector_store(lang)

    def _get_store_path(self, language: str) -> str:
        """Get the file path for a language's vector store."""
        return os.path.join(self.persist_dir, f"{language}_index")
    
    def _load_vector_store(self, language: str) -> None:
        """Load a vector store for a specific language if it exists."""
        store_path = self._get_store_path(language)
        if os.path.exists(store_path):
            try:
                self.vector_stores[language] = FAISS.load_local(
                    store_path,
                    self.embedder,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing vector store for {language}")
            except Exception as e:
                logger.error(f"Error loading vector store for {language}: {e}")
    
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
            
            lang_docs = []

            # Store Document of each chunk in the FAISS index
            for i, doc in enumerate(chunked_docs.get(lang, [])):
                logger.info(f"Chunk ID: {doc['chunk_id']}, Doc ID: {doc['doc_id']}, Text: {doc['text'][:50]}...")
                metadata = {
                    "chunk_id": doc["chunk_id"],
                    "doc_id": doc["doc_id"],
                    "chunk_idx": doc["chunk_idx"],
                    "parallel_id": doc.get("parallel_id", None)
                }
                lang_docs.append(
                    Document(
                        page_content=doc["text"],
                        metadata=metadata
                    )
                )
            
            if not lang_docs:
                logger.warning(f" No documents to add for {lang}.")
                continue

            if self.vector_stores[lang] is None:
                # Create new FAISS index
                self.vector_stores[lang] = FAISS.from_documents(
                    documents=lang_docs,
                    embedding=self.embedder,
                )
            else:
                # Add to existing index
                self.vector_stores[lang].add_documents(lang_docs)
            
            store_path = self._get_store_path(lang)
            self.vector_stores[lang].save_local(store_path)
            logger.info(f"Updated and saved vector store for {lang}")

    def search(self, query: str, language: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents in the specified language.
        
        Args:
            query: The search query
            language: The language to search in
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if language not in self.vector_stores or self.vector_stores[language] is None:
            raise ValueError(f"No vector store available for {language}")
        
        return self.vector_stores[language].similarity_search(query, k=k)
    
    

            

