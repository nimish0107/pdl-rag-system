import pytesseract
from PIL import Image

def read_image(image_path):
    """
    Reads an image from a given path using the Pillow library.
    Returns the opened image object.
    """

    return Image.open(image_path)
    
    
def image_to_text(image):
    """
    Convert an image to text using Tesseract OCR.

    Args:
        image (PIL.Image): Opened image file using PIL.

    Returns:
        str: Extracted text from the image.
    """

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image, lang="pan")

    return text