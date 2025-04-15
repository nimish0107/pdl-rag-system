from PIL import Image
from OCR.ocr import image_to_text
from OCR.ocr import read_image

print("Image reading and OCR processing script started.")
image_path = r"N:\Coding\Projects\PDL\data\344.jpg"  # Replace with your image path
image = read_image(image_path)
text = image_to_text(image_path)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Extracted text from the image:")
print(text)