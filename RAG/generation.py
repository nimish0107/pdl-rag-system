import os
import json
import requests
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class OllamaAnswerGenerator:
    """
    Class to generate answers from retrieved documents using locally hosted Ollama LLM.
    """
    
    def __init__(self, model_name: str = "gemma3", ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the answer generator with an Ollama model.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_base_url: Base URL for Ollama API
        """
        # Configure streaming callbacks for real-time output
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Initialize Ollama LLM integration
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            callback_manager=callback_manager,
            temperature=0.1,  # Lower temperature for more deterministic answers
            verbose=True
        )
        
        # Define prompt templates for different languages
        self.prompt_templates = {
            "punjabi": PromptTemplate(
                input_variables=["query", "context"],
                template="""
                ਹੇਠ ਦਿੱਤੇ ਦਸਤਾਵੇਜ਼ਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਦਿੱਤੇ ਗਏ ਸਵਾਲ ਦਾ ਉੱਤਰ ਦਿਓ।
                
                ਸਵਾਲ: {query}
                
                ਦਸਤਾਵੇਜ਼:
                {context}
                
                ਉੱਤਰ:
                """
            ),
            "hindi": PromptTemplate(
                input_variables=["query", "context"],
                template="""
                नीचे दिए गए दस्तावेजों के आधार पर दिए गए प्रश्न का उत्तर दें।
                
                प्रश्न: {query}
                
                दस्तावेज:
                {context}
                
                उत्तर:
                """
            ),
            "english": PromptTemplate(
                input_variables=["query", "context"],
                template="""
                Answer the question based on the documents provided below.
                
                Question: {query}
                
                Documents:
                {context}
                
                Answer:
                """
            )
        }
        
        # Create LLM chains for each language
        self.chains = {
            lang: LLMChain(llm=self.llm, prompt=prompt)
            for lang, prompt in self.prompt_templates.items()
        }
    
    def check_ollama_availability(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(self.llm.base_url + "/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def format_documents_for_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a string context for the LLM prompt.
        
        Args:
            documents: List of Document objects retrieved from vector store
            
        Returns:
            Formatted context string
        """
        context_str = ""
        
        for i, doc in enumerate(documents):
            context_str += f"Document {i+1}:\n{doc.page_content}\n\n"
        
        return context_str
    
    def generate_answer(self, query: str, documents: List[Document], language: str) -> str:
        """
        Generate an answer for the given query based on retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            language: Language of the query and documents
            
        Returns:
            Generated answer
        """
        if language not in self.chains:
            raise ValueError(f"Unsupported language: {language}")
        
        if not documents:
            return "No relevant documents found to answer your question."
        
        # Check if Ollama is available
        if not self.check_ollama_availability():
            return "Error: Could not connect to the Ollama server. Please ensure it's running."
        
        # Format documents into context string
        context = self.format_documents_for_context(documents)
        
        # Generate answer using the appropriate chain
        try:
            response = self.chains[language].run(query=query, context=context)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"