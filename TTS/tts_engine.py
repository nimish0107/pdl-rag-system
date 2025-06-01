from gtts import gTTS
from utils import valid_languages
from utils import logger
import io
def generate_audio(text: str, language: str):
    """Generate audio from text using gTTS"""
    if not text.strip():
        logger.error("Text cannot be empty.")
        return None
        
    if language not in valid_languages:
        logger.error(f"Invalid language. Supported languages: {', '.join(valid_languages.keys())}")
        return None

    try:
        # Create TTS object
        tts = gTTS(text=text, lang=valid_languages[language], slow=False)
        
        # Create buffer to store audio
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except ValueError as ve:
        logger.error(f"TTS Error: {str(ve)}")
        return None
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None