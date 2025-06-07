FROM python:3.10-slim

# --- System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-pan \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- Set Hugging Face cache
ENV TRANSFORMERS_CACHE=/models/huggingface
RUN mkdir -p /models/huggingface

# --- App directory
WORKDIR /app

# --- Copy source
COPY . .

# --- Install Python dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# --- Preload Hugging Face models
RUN python models.py

# --- Pull Gemma model and create pdlRAG model
# RUN ollama pull gemma:3b && ollama create pdlRAG -f MODELFILE
# RUN bash -c "ollama serve; \
# ollama pull gemma:3b; ollama create pdlRAG -f ./Modelfile"
RUN bash -c "\
    # 8.1 Start a temporary Ollama server in the background \
    echo '⏳ Starting ephemeral Ollama server…'; \
    ollama serve & \
    TEMP_PID=\$!; \
    # 8.2 Give it a few seconds to fully initialize and bind port 11434 \
    sleep 5; \
    # 8.3 Pull the gemma3 model (blocking until download completes) \
    echo '⏳ Pulling gemma3…'; \
    ollama pull gemma3; \
    # 8.4 Create your custom pdlRAG from MODELFILE \
    echo '⏳ Creating pdlRAG model…'; \
    ollama create pdlRAG -f ./Modelfile; \
    # 8.5 Stop the temporary Ollama server \
    echo '🛑 Stopping ephemeral Ollama server…'; \
    kill \$TEMP_PID; \
    echo '✅ gemma3 + pdlRAG baked into image.' \
"
# --- Expose relevant ports
EXPOSE 11434 8000 8501

RUN ollama create pdlRAG -f ./Modelfile

# --- Entrypoint
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
