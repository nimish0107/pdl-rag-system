# 🔍 Punjabi Document Language (PDL) - RAG System

This project is focused on building a **Retrieval-Augmented Generation (RAG)** system that enables intelligent Q&A in **Punjabi, Hindi, and English** languages. It translates Punjabi documents, stores multilingual versions in a database, and allows language-specific querying with both **text and speech responses**.

---

## 🚀 Features

- Upload Punjabi documents and automatically translate them into English and Hindi.
- Store all language versions in respective vector databases.
- Multilingual RAG pipeline to ask questions in your preferred language.
- Generate answers in **text** and **speech**.
- Evaluate translation model performance using BLEU and METEOR scores.
- Fully CPU-compatible pipeline.

---

## 🧠 System Architecture

### 1. Document Upload
- Upload Punjabi documents.
- Translate them to English and Hindi.
- Vectorize and store in dedicated language tables.

### 2. RAG System
- Select a language for interaction.
- Query the corresponding language table.
- Retrieve relevant documents using vector search.
- Generate an answer using a language-specific prompt.
- Return the answer in both text and audio formats.

### 3. Testing
- Automatically generate QA pairs from documents.
- Evaluate pipeline performance using standard metrics.

---

## 🔤 Translation Model Benchmarking

---

## 📁 Tech Stack

1. OCR -> TesseractOCR
2. Translation -> IndicTrans2
3. RAG -> Langchain
4. Embeddings -> intfloat/e5-small
5. LLM -> Ollama, Gemma3
6. Vector Store -> FAISS

---
## 📚 Datasets Used
The dataset was sourced from Panjab Digital Library (PDL) and consists of scanned images
of historical books, manuscripts, and archival materials. These documents span various
genres, including religious texts, literary works, and administrative records, and are primarily
written in Gurmukhi script. The images were later processed for text extraction and language
translation tasks.

---

## 🧪 Evaluation Metrics

- **BLEU Score**
- **METEOR Score**

Separate evaluations are done for:
- Punjabi → English
- Punjabi → Hindi

---

## 📁 Setting Up
1. install all requirements using command: <br> ```pip install -r requirements.txt```
2. setup tesseract-ocr and other dependencies using [Tesseract Setup](./OCR/README.md)
3. setup ollama and related dependencies using [Ollama Setup](./RAG/settingOllama.md)
4. create ollama model pdlRAG using command: <br>
    ```ollama create pdlRAG -f Modelfile```
5. install all required models using ```models.py```:<br>
    Run Command ```python models.py```
---
## 👨‍💻 Running the project
1. Run Ollama server - use command: <br>
    ```ollama serve```
2. Open a new terminal
3. For Fast API, run using command:<br>
    ```uvicorn main:app```<br>
    For Streamlit UI, run using command: <br>
    ```streamlit run app.py```
---

## 👥 Contributors

- **Harshita** – Punjabi to English translation
- **Nimish** – Punjabi to Hindi translation

---

## 📌 Notes

- The entire system is designed to run on **CPU** for broader accessibility and deployment.
- Target languages: **pa** (Punjabi), **en** (English), **hi** (Hindi)

---

