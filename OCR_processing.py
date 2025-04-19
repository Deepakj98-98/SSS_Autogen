import pytesseract
import re
from pdf2image import convert_from_bytes
import os
import shutil
from PIL import Image
from docx import Document

class OCR_processing:
    def ocr_pdf_text(self,file_buffer):
         final_text=[]
         pytesseract.tessseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
         output_dir="frames"
         pop_path=r'C:/Users/Deepak J Bhat/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin'

         file_buffer.seek(0)
         
         images=convert_from_bytes(file_buffer.read(), poppler_path=pop_path, fmt='jpeg')

         for img in images:
          text=pytesseract.image_to_string(img)
          final_text.append(text)

         ocr_text="".join(final_text)
         return ocr_text

    def pdf_text(self,file_buffer):
         #output_dir="frames"
         file_buffer.seek(0)
         ocr_text=self.ocr_pdf_text(file_buffer)
         
         cleaned_text = re.sub(r'[^A-Za-z0-9\s.,]', '', ocr_text)
         cleaned_text1 = re.sub(r'[^\x20-\x7E\n]', '', cleaned_text)
         return cleaned_text1
    
    def doc_text(self, file_buffer):
         file_buffer.seek(0)
         doc=Document(file_buffer)
         final_text = [para.text for para in doc.paragraphs if para.text.strip()]
         doc_text = " ".join(final_text)
         cleaned_text = re.sub(r'[^A-Za-z0-9\s.]', '', doc_text)
         text = re.sub(r'[^\w\s.,!?]', '', cleaned_text)
         cleaned_text1 = re.sub(r'[^\x20-\x7E\n]', '', text)
         return cleaned_text1

