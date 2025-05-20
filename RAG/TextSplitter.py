from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any
import hashlib

LANGUAGES = ["punjabi", "hindi", "english"]
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

class MultilingualTextSplitter:
    """Custom text splitter that maintains parallel chunks across languages."""
    
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Base splitter for creating initial chunks
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "।", ".", "!", "?"]
        )
    
    def generate_chunk_id(self, text: str, doc_id: str, idx: int) -> str:
        """Generate a deterministic ID for a chunk based on content and position."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        return f"{doc_id}_{idx}_{content_hash}"
    
    def split_documents(self, documents: Dict[str, List[str]], doc_uuid: str) -> Dict[str, List[Dict]]:
        """
        Split documents in all languages while maintaining alignment.
        
        Args:
            documents: Dictionary mapping language to list of documents
            doc_uuid: Unique identifier for the document set
            
        Returns:
            Dictionary mapping language to list of chunk dictionaries
        """
        result = {lang: [] for lang in LANGUAGES}
        
        # First, chunk the Punjabi documents (source language)
        punjabi_chunks = []
        for doc_idx, doc_text in enumerate(documents["punjabi"]):
            doc_id = f"doc_{doc_uuid}"
            chunks = self.base_splitter.create_documents([doc_text])
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = self.generate_chunk_id(chunk.page_content, doc_id, chunk_idx)
                punjabi_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_idx": chunk_idx,
                    "text": chunk.page_content
                })
        
        # Store Punjabi chunks
        result["punjabi"] = punjabi_chunks
        
        # Now create parallel chunks for Hindi and English
        for lang in ["hindi", "english"]:
            # We'll assume we have parallel documents in the same order
            for doc_idx, doc_text in enumerate(documents[lang]):
                doc_id = f"doc_{doc_uuid}"
                # Use the same chunking strategy to maintain roughly parallel chunks
                chunks = self.base_splitter.create_documents([doc_text])
                
                # Match each chunk with its Punjabi counterpart
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk_idx < len(punjabi_chunks) and punjabi_chunks[chunk_idx]["doc_id"] == doc_id:
                        result[lang].append({
                            "doc_id": doc_id,
                            "chunk_id": f"{lang}_{punjabi_chunks[chunk_idx]['chunk_id']}",
                            "parallel_id": punjabi_chunks[chunk_idx]["chunk_id"],
                            "chunk_idx": chunk_idx,
                            "text": chunk.page_content
                        })
        
        return result