import asyncio
from PIL import Image
from OCR.ocr import image_to_text
from OCR.ocr import read_image
from Translation.translate import translate_punjabi_to_HindiEnglish
from utils import logger
from RAG.TextSplitter import MultilingualTextSplitter, CHUNK_OVERLAP, CHUNK_SIZE
from RAG.embeddings import FaissEmbeddingStore
import os
text_splitter = MultilingualTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
store = FaissEmbeddingStore()

logger.info("Image reading and OCR processing script started.")
image_path = r"N:\Coding\Projects\PDL\data\344.jpg"  # Replace with your image path

image = read_image(image_path)
text = image_to_text(image)
logger.info("OCR processing completed.")

logger.info("Translating text...")
translated_text = asyncio.run(translate_punjabi_to_HindiEnglish(text))
logger.info("Translation completed.")
# output_text = f"English:\n{translated_text['english']}\n\nHindi:\n{translated_text['hindi']}\n\nPunjabi:\n{translated_text['punjabi']}"
# with open("output.txt", "w", encoding="utf-8") as f:
#     f.write(output_text)
# logger.info("Translated text from the image:")
# logger.info(output_text)
logger.info("Text chunking started.")
chunked_docs = text_splitter.split_documents(translated_text)
logger.info("Text chunking completed.")



logger.info(chunked_docs)

embeddings = store.add_documents(chunked_docs)


logger.info("Embeddings generated and stored successfully.")
# store.save_all()
logger.info("Embeddings stored successfully.")
logger.info("Pipeline completed successfully.")


# import asyncio
# import gradio as gr
# from PIL import Image
# from OCR.ocr import image_to_text
# from OCR.ocr import read_image
# from Translation.translate import translate_punjabi_to_HindiEnglish
# from utils import logger
# import os

# # Function that will process the input image and return the output text
# def process_image(image: Image.Image):
#     logger.info("Image reading and OCR processing started.")
    
#     text = image_to_text(image)
#     logger.info("OCR processing completed.")

#     logger.info("Translating text...")
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     translated_text = loop.run_until_complete(translate_punjabi_to_HindiEnglish(text))
#     logger.info("Translation completed.")

#     output_text = (
#         f"English:\n{translated_text['english']}\n\n"
#         f"Hindi:\n{translated_text['hindi']}\n\n"
#         f"Punjabi:\n{translated_text['punjabi']}"
#     )
    
#     logger.info("Pipeline completed successfully.")
#     return output_text

# # Gradio interface for the app
# iface = gr.Interface(
#     fn=process_image, 
#     inputs=gr.Image(type="pil", label="Upload Image"), 
#     outputs=gr.Textbox(label="Translated Text"), 
#     live=False  # Optional: Set to True for live updates (useful for larger models)
# )

# # Launch the Gradio app
# if __name__ == "__main__":
#     iface.launch()
