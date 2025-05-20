# 🧠 Ollama Setup Guide
Ollama lets you run open-source large language models (LLMs) locally with ease. This guide helps you set up Ollama on your system.

## 📦 Requirements
+ Operating System: macOS (M1/M2 and Intel), Linux (x86_64), or Windows (via WSL2)

+ Disk Space: At least 8–20 GB (varies depending on model)

+ Memory: 16+ GB recommended

+ Internet: Required for downloading models

## 🚀 Installation
### 🔧 macOS
Install via shell script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Or install via Homebrew:
```bash
brew install ollama
```
### 🐧 Linux
Install via shell script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Make sure Docker is running if prompted.

### 🪟 Windows (via WSL2)
Install WSL2 and a Linux distro (e.g., Ubuntu).

Open your WSL terminal and run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## ✅ Verifying the Installation
After installation, run:
```bash
ollama --version
```
You should see the installed version printed.

## 🤖 Running a Model
To pull and run a model (e.g., Gemma3 for our use case):
```bash
ollama run gemma3
```
Ollama will automatically download the model if it’s not already available locally.

To pull a model only
```bash
ollama pull gemma3
```
## 📚 Available Models
You can find available models at: https://ollama.com/library

## Serve via API
```bash
ollama serve
```
Use the REST API at http://localhost:11434.