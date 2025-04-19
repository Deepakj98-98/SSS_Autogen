import os
from OCR_processing import OCR_processing
from image_text_extraction import Image_text_Extraactor
from langchain_trial import Chatbot_response
from qdrant_chunking import QdrantChunking
from qdrant_retrieval import Qdrant_retrieval
from whisper_transcripts import Whisper_transcripts
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify,send_from_directory
import speech_recognition as sr
import asyncio
from pymongo import MongoClient
from gridfs import GridFSBucket

class mongoDbTrial:
    def __init__(self,folder):
        self.upload_folder=folder
        os.makedirs(folder, exist_ok=True)
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db=self.client["file_db"]
        self.fs_bucket=GridFSBucket(self.db)

    def save_file(self,files):
        uploaded_file_paths = []
        for filepath in files:
            filename = os.path.basename(filepath)
            with open(filepath, "rb") as f:
                data = f.read()

            with self.fs_bucket.open_upload_stream(
                filename,
                chunk_size_bytes=1048576,
                metadata={"source": "local_upload"}
            ) as grid_in:
                grid_in.write(data)

            uploaded_file_paths.append(filepath) # Handles binary files correctly

        print(uploaded_file_paths)
    
    def retreive_files(self):
        for file_doc in self.fs_bucket.find({}):
            print(file_doc)
        
    def print_all_res(self):
        # List all files
        for grid_out in self.fs_bucket.find():
            print(f"\nFilename: {grid_out.filename}")
            if grid_out.filename.endswith(".txt"):
                content = grid_out.read().decode("utf-8")  # decode bytes to string
                print("File content:")
                print(content)

mongo_db=mongoDbTrial("demo")
folder="demo"
files_to_upload = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
mongo_db.print_all_res()