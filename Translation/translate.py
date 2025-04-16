import torch
import asyncio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.indictranstoolkit.IndicTransToolkit.processor import IndicProcessor
DEVICE = "cpu"
print(f"[INFO] Using device: {DEVICE}")


src_lang = "pan_Guru"
model_name_en = "ai4bharat/indictrans2-indic-en-dist-200M"
model_name_hi = "ai4bharat/indictrans2-indic-indic-dist-320M"


tokenizer_en = AutoTokenizer.from_pretrained(model_name_en, trust_remote_code=True)
model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en, trust_remote_code=True).to(DEVICE)

tokenizer_hi = AutoTokenizer.from_pretrained(model_name_hi, trust_remote_code=True)
model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name_hi, trust_remote_code=True).to(DEVICE)

ip = IndicProcessor(inference=True)

async def translate_batch(batch, src, tgt, tokenizer, model):
    preprocessed = ip.preprocess_batch(batch, src_lang=src, tgt_lang=tgt)
    inputs = tokenizer(preprocessed, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1000,
            num_beams=5,
            num_return_sequences=1,
        )
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=tgt)

async def translate_punjabi_to_HindiEnglish(input_text):
    
    input_sentences = input_text.split("\n\n")

    task_en = translate_batch(input_sentences, src_lang, "eng_Latn", tokenizer_en, model_en)
    task_hi = translate_batch(input_sentences, src_lang, "hin_Deva", tokenizer_hi, model_hi)

    translations_en, translations_hi = await asyncio.gather(task_en, task_hi)

    output_lines = {"punjabi": "", "english": "", "hindi": ""}
    for punjabi, en, hi in zip(input_sentences, translations_en, translations_hi):
        output_lines["punjabi"] += punjabi + "\n\n"
        output_lines["english"] += en + "\n\n"
        output_lines["hindi"] += hi + "\n\n"

    # with open("translations_output.txt", "w", encoding="utf-8") as f:
    #     f.writelines(output_lines)

    print("\nTranslation Completed.")
    return output_lines

if __name__ == "__main__":
    
    input_sentences = [
        "ਜਦੋਂ ਮੈਂ ਛੋਟਾ ਸੀ, ਮੈਂ ਹਰ ਰੋਜ਼ ਪਾਰਕ ਜਾਂਦਾ ਸੀ।",
        "ਅਸੀਂ ਪਿਛਲੇ ਹਫ਼ਤੇ ਇੱਕ ਨਵੀਂ ਫਿਲਮ ਵੇਖੀ ਜੋ ਬਹੁਤ ਪ੍ਰੇਰਣਾਦਾਇਕ ਸੀ।",
        "ਜੇਕਰ ਤੁਸੀਂ ਮੈਨੂੰ ਉਸ ਸਮੇਂ ਮਿਲਦੇ, ਤਾਂ ਅਸੀਂ ਬਾਹਰ ਖਾਣਾ ਖਾਣੇ ਜਾਂਦੇ।",
        "ਮੇਰੇ ਦੋਸਤ ਨੇ ਮੈਨੂੰ ਉਸਦੀ ਜਨਮਦਿਨ ਦੀ ਪਾਰਟੀ ਵਿੱਚ ਬੁਲਾਇਆ ਹੈ, ਅਤੇ ਮੈਂ ਉਸਨੂੰ ਇੱਕ ਤੋਹਫਾ ਦੇਵਾਂਗਾ।",
    ]
    output = asyncio.run(translate_punjabi_to_HindiEnglish(input_sentences))
    print(output)