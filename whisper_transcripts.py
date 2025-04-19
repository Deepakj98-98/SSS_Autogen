import whisper
from pydub import AudioSegment
from pydub.utils import make_chunks
import spacy
import warnings
import re
import os
import shutil
from io import BytesIO
import tempfile

class Whisper_transcripts:
    def __init__(self, spacy_model="en_core_web_sm"):
        warnings.filterwarnings("ignore",category=FutureWarning)
        self.nlp=spacy.load(spacy_model)
        self.whisper_model=whisper.load_model("base")
    
    def transcribe(self,file_buffer,extension):
        file_buffer.seek(0)
        audio=AudioSegment.from_file(file_buffer, format=extension.lstrip("."))
        chunks=make_chunks(audio, 30000)
        
        transcript=""
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                chunk.export(tmpfile.name, format="wav")
                result = self.whisper_model.transcribe(tmpfile.name)
                transcript += result["text"] + " "

        return transcript
    
    def process_audio_video(self, file_buffer,extension):
        text=self.transcribe(file_buffer,extension)
        text=re.sub(r'\s+',' ',text)
        text=re.sub(r'[^\w\s,!?]','',text)
        text=re.sub(r'([.,?])\1+',r'\1',text)

        return text