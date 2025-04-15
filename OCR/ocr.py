import pytesseract
from PIL import Image

def read_image(image_path):
    """
    Reads an image from a given path using the Pillow library.
    Returns the opened image object.
    """
    with Image.open(image_path) as img:
        return img
    
    
def image_to_text(image):
    """
    Convert an image to text using Tesseract OCR.

    Args:
        image (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """

    # Load the image from the specified path
    img = Image.open(image)

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(img, lang="pan")

    return text