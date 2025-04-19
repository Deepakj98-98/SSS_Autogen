import pytesseract
import re
import cv2
import numpy as np

class Image_text_Extraactor:
    def image_extraction(self,file_buffer):
        file_buffer.seek(0)
        file_bytes = np.frombuffer(file_buffer.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from buffer.")
        pytesseract.tessseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        
        grayscale=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        histogram_equalization=cv2.equalizeHist(grayscale)

        gaussian_blur=cv2.GaussianBlur(histogram_equalization,(5,5),0)

        _,binary=cv2.threshold(gaussian_blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )

        text=pytesseract.image_to_string(binary)
        cleaned_text = re.sub(r'[^A-Za-z0-9\s.]', '', text)
        cleaned_text = re.sub(r'[^\x20-\x7E\n]', '', cleaned_text)
        return cleaned_text