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

## 📚 Datasets Used

---

## 🧪 Evaluation Metrics

- **BLEU Score**
- **METEOR Score**

Separate evaluations are done for:
- Punjabi → English
- Punjabi → Hindi

---

## 📁 Tech Stack

---

## 👥 Contributors

- **Harshita** – Punjabi to English translation
- **Nimish** – Punjabi to Hindi translation

---

## 📝 To Do


---

## 📌 Notes

- The entire system is designed to run on **CPU** for broader accessibility and deployment.
- Target languages: **pa** (Punjabi), **en** (English), **hi** (Hindi)

---

