from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
translate_models = [
    "ai4bharat/indictrans2-indic-en-dist-200M",  # Replace/add your models
    "ai4bharat/indictrans2-indic-indic-dist-320M"
]
embedding_models = [
    "intfloat/multilingual-e5-small"
]
for model in translate_models:
    print(f"Preloading: {model}")
    AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    AutoModelForSeq2SeqLM.from_pretrained(model, trust_remote_code=True)

for model in embedding_models:
    print(f"Preloading: {model}")
    SentenceTransformer(model, trust_remote_code=True)