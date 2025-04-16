import asyncio
from PIL import Image
from OCR.ocr import image_to_text
from OCR.ocr import read_image
from Translation.translate import translate_punjabi_to_HindiEnglish

print("Image reading and OCR processing script started.")
image_path = r"N:\Coding\Projects\PDL\data\344.jpg"  # Replace with your image path
image = read_image(image_path)
text = image_to_text(image_path)
print("OCR processing completed.")

print("Translating text...")
translated_text = asyncio.run(translate_punjabi_to_HindiEnglish(text))
print("Translation completed.")
output_text = f"English:\n{translated_text['english']}\n\nHindi:\n{translated_text['hindi']}\n\nPunjabi:\n{translated_text['punjabi']}"
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)
print("Translated text from the image:")
print(output_text)