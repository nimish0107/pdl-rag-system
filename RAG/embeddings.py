from langchain_community.vectorstores import FAISS
import numpy as np
import os
import shutil
import json
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from utils import logger  
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from utils import DATA_DIR

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
        texts = ["passage: " + text for text in texts]  # Prepend 'passage: ' to each text
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query using the SentenceTransformer model.
        """
        query = "query: " + query  # Prepend 'query: ' to the query
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
    
    
    def delete_document_by_id(self, doc_id_file: str):
        """
        Delete all vectors associated with the given doc_id from each language's FAISS store.
        """
        doc_id = f"doc_{doc_id_file}"  # Ensure doc_id is formatted correctly
        deleted_any = False
        for lang in LANGUAGES:
            store = self.vector_stores.get(lang)
            if store is None:
                logger.warning(f"No vector store found for language: {lang}")
                continue

            # Fetch all existing documents
            existing_docs = store.docstore._dict.values()
            remaining_docs = [doc for doc in existing_docs if doc.metadata.get("doc_id") != doc_id]

            if len(remaining_docs) == len(existing_docs):
                logger.info(f"No documents found for deletion with doc_id: {doc_id} in {lang}")
                continue
            
            non_empty_docs = [doc for doc in remaining_docs if doc.page_content.strip()]
            if not non_empty_docs:
                deleted_any = True
                logger.warning(f"All remaining docs for {lang} are empty. Skipping FAISS index rebuild.")
                self.vector_stores[lang] = None
                if os.path.exists(self._get_store_path(lang)):
                    shutil.rmtree(self._get_store_path(lang))
                continue
            logger.info(f"Rebuilding FAISS index for {lang} without doc_id: {doc_id}")

            if len(remaining_docs) < len(existing_docs):
                deleted_any = True
                new_index = FAISS.from_documents(
                    documents=remaining_docs,
                    embedding=self.embedder
                )
                self.vector_stores[lang] = new_index
                new_index.save_local(self._get_store_path(lang))
            logger.info(f"Updated vector store saved after deletion for {lang}")

        for ext in ["jpg", "jpeg", "png"]:
            file_path = os.path.join(DATA_DIR, f"{doc_id_file}.{ext}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                    deleted_any = True
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")
        return deleted_any
    
    def delete_all_documents(self) -> bool:
        """
        Deletes all documents and clears all FAISS indexes for all languages.
        """
        deleted = False
        for lang in LANGUAGES:
            if self.vector_stores[lang] is not None:
                self.vector_stores[lang] = None  # Clear from memory
                index_path = self._get_store_path(lang)
                if os.path.exists(index_path):
                    shutil.rmtree(index_path)  # Delete the directory with index.faiss and index.pkl
                    logger.info(f"Deleted FAISS index directory for {lang}")
                    deleted = True
                if os.path.exists(DATA_DIR):
                    shutil.rmtree(DATA_DIR)
                    os.makedirs(DATA_DIR, exist_ok=True)
        return deleted