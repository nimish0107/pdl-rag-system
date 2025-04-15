## Installation Steps

### Tesseract
+ Windows:
    
    1. Download and install Tesseract from the official site or via Chocolatey:
        `choco install tesseract`
    2. Go to official github of tesseract-ocr and download traineddata for the desired language i.e Punjabi

        `pan.traineddata` should be downloaded
    
    Add this to `C:\Program Files\Tesseract-OCR\tessdata` if Tesseract-OCR is installed in C:\Program Files
+ Linux:
    1. Use your package manager:
         sudo apt-get install tesseract-ocr tesseract-ocr-pan

### Python Requirements
• Install Pillow and Pytesseract:
    pip install pillow pytesseract