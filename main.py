import asyncio
from PIL import Image
from OCR.ocr import image_to_text
from OCR.ocr import read_image
from Translation.translate import translate_punjabi_to_HindiEnglish
from utils import logger

logger.info("Image reading and OCR processing script started.")
image_path = r"N:\Coding\Projects\PDL\data\344.jpg"  # Replace with your image path
image = read_image(image_path)
text = image_to_text(image_path)
logger.info("OCR processing completed.")

logger.info("Translating text...")
translated_text = asyncio.run(translate_punjabi_to_HindiEnglish(text))
logger.info("Translation completed.")
output_text = f"English:\n{translated_text['english']}\n\nHindi:\n{translated_text['hindi']}\n\nPunjabi:\n{translated_text['punjabi']}"
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)
logger.info("Translated text from the image:")
logger.info(output_text)
logger.info("Pipeline completed successfully.")