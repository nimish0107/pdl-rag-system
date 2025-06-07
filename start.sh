#!/bin/bash

# Start Ollama
ollama serve &
sleep 5
# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
streamlit run app.py --server.port 8501 --server.headless true

# Wait to keep container running
wait
