import torch
import os
import logging
import soundfile as sf
from typing import Optional, Tuple
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import numpy as np
import gc
from utils import logger
# Language-specific voice descriptions
VOICE_DESCRIPTIONS = {
    "english": " Priya speaks at a moderate pace, clear and neutral tone, high-quality recording with no background noise.",
    "hindi": "Rohit speaks at a moderate pace, moderate pace, clear and neutral tone, high-quality recording with no background noise.",
    "punjabi": "Divjot  speaks at a moderate pace, clear and neutral tone, high-quality recording with no background noise."
}

def load_model_and_tokenizers(model_name: str = "ai4bharat/indic-parler-tts") -> Tuple[
    Optional[ParlerTTSForConditionalGeneration], Optional[AutoTokenizer], Optional[AutoTokenizer]
]:
    """
    Load the TTS model and tokenizers with quantization for CPU optimization.
    """
    device = "cpu"
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        # Dynamic quantization (only applies to torch.nn.Linear, not whole ParlerTTS)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("Applied dynamic quantization for CPU optimization")

        return model, tokenizer, description_tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or tokenizers: {e}")
        return None, None, None

def cleanup_resources(model: Optional[ParlerTTSForConditionalGeneration], *args) -> None:
    """
    Clean up model and other resources to free memory.
    """
    try:
        if model is not None:
            del model
        for arg in args:
            if arg is not None:
                del arg
        gc.collect()
        torch.cuda.empty_cache()  # Just in case it’s ever run on GPU
        logger.info("Cleaned up model and tokenizer resources")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def synthesize_speech(
    prompt: str,
    language: str,
    output_path: str = "indic_tts_out.wav",
    voice_description: Optional[str] = None,
    keep_resources: bool = False,
    model_name: str = "ai4bharat/indic-parler-tts"
) -> Optional[np.ndarray]:
    """
    Generate TTS audio for a given prompt in English, Hindi, or Punjabi with an Indian accent.
    """
    if not prompt or not isinstance(prompt, str):
        logger.error("Prompt must be a non-empty string")
        return None
    language = language.lower()
    if language not in VOICE_DESCRIPTIONS:
        logger.error(f"Unsupported language: {language}. Choose 'english', 'hindi', or 'punjabi'.")
        return None

    voice_description = voice_description or VOICE_DESCRIPTIONS[language]

    output_dir = os.path.dirname(output_path) or "."
    if not os.path.exists(output_dir):
        logger.error(f"Output directory {output_dir} does not exist")
        return None
    if os.path.exists(output_path):
        logger.warning(f"Output file {output_path} already exists and will be overwritten")

    model, tokenizer, description_tokenizer = load_model_and_tokenizers(model_name)
    if model is None or tokenizer is None or description_tokenizer is None:
        logger.error("Failed to initialize model or tokenizers")
        return None

    try:
        description_inputs = description_tokenizer(voice_description, return_tensors="pt").to("cpu")
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

        with torch.no_grad():
            generation = model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )

        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_path, audio_arr, model.config.sampling_rate)
        logger.info(f"Saved TTS output to {output_path}")
        return audio_arr

    except Exception as e:
        logger.error(f"Error during speech synthesis: {e}")
        return None

    finally:
        if not keep_resources:
            cleanup_resources(model, tokenizer, description_tokenizer)
        else:
            logger.info("Keeping model and tokenizers in memory for reuse")

# Example usage
if __name__ == "__main__":
    prompts = {
        "english": "Hello, how are you !",
        "hindi": "नमस्ते, इंडिक पार्लर टीटीएस डेमो में आपका स्वागत है!",
        "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇੰਡਿਕ ਪਾਰਲਰ ਟੀਟੀਐਸ ਡੈਮੋ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ!"
    }

    for i, (lang, prompt) in enumerate(prompts.items()):
        output_file = f"output_{lang}.wav"
        keep = i < len(prompts) - 1  # Keep resources for all but the last
        audio = synthesize_speech(
            prompt=prompt,
            language=lang,
            output_path=output_file,
            keep_resources=keep
        )
        if audio is not None:
            logger.info(f"Successfully generated {lang} audio: {output_file}")
