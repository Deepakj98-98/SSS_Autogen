import os
from OCR_processing import OCR_processing
from image_text_extraction import Image_text_Extraactor
from langchain_trial import run_chatbot
from qdrant_chunking import QdrantChunking
from qdrant_retrieval import Qdrant_retrieval
from whisper_transcripts import Whisper_transcripts
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify,send_from_directory,abort,Response
import speech_recognition as sr
import asyncio
from pymongo import MongoClient
from gridfs import GridFSBucket
import mimetypes
import io



class FileProcessor:
    def __init__(self, folder,mongo_client, fs_bucket,db):
        self.upload_folder=folder
        os.makedirs(folder, exist_ok=True)
        self.ocr=OCR_processing()
        self.image_processing=Image_text_Extraactor()
        self.transcripts=Whisper_transcripts()
        self.client = mongo_client
        self.db=db
        self.fs_bucket=fs_bucket
    
    def save_file(self, files):
        uploaded_file_paths=[]
        for file in files:
            filename=file.filename
            file.seek(0)
            with self.fs_bucket.open_upload_stream(
                filename,
                chunk_size_bytes= 1048576,
                metadata={"source":"direct_upload"}
            ) as grid_in:
                grid_in.write(file.read())
            uploaded_file_paths.append(filename)
        return uploaded_file_paths
            
    def process_files(self,filenames):
        for filename in filenames:
            text = ""
            extension = os.path.splitext(filename)[1].lower()

            # Download the file from GridFS to memory for processing
            grid_out = self.fs_bucket.open_download_stream_by_name(filename)
            file_data = grid_out.read()

            file_buffer = io.BytesIO(file_data)

            # Process based on file type
            if extension in (".mp4", ".mp3", ".wav"):
                text += self.transcripts.process_audio_video(file_buffer,extension)
            elif extension in (".doc", ".docx"):
                text += self.ocr.doc_text(file_buffer)
            elif extension == ".pdf":
                text += self.ocr.pdf_text(file_buffer)
            elif extension in (".png", ".jpg", ".jpeg"):
                text += self.image_processing.image_extraction(file_buffer)

            # Save text back to GridFS
            self.save_text_file(filename, text)

            # Optionally remove temp file
            #os.remove(temp_file_path)

    def save_text_file(self, original_filename, text):
        filename_no_ext = os.path.splitext(original_filename)[0]
        text_file = f"{filename_no_ext}.txt"

        with self.fs_bucket.open_upload_stream(
            text_file,
            chunk_size_bytes=1048576,
            metadata={"source": "generated_text"}
        ) as grid_in:
            grid_in.write(text.encode("utf-8"))  # Write text as bytes

app=Flask(__name__)
UPLOAD_FOLDER = "uploads"

qdrant_chunking = QdrantChunking()
collection_name="test_check1"
query_processor=Qdrant_retrieval(collection_name)
r = sr.Recognizer()
conversations={}
client = MongoClient("mongodb://localhost:27017/")
db=client["file_db"]
fs_bucket=GridFSBucket(db)
file_processor = FileProcessor(UPLOAD_FOLDER,mongo_client=client,db=db,fs_bucket=fs_bucket)


@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        files=request.files.getlist("files")
        uploaded_file_paths=file_processor.save_file(files)
        if uploaded_file_paths:
            file_processor.process_files(uploaded_file_paths)
            qdrant_chunking.builder_graph(fs_bucket,collection_name)
        return redirect(url_for("home"))
    #folders=UPLOAD_FOLDER
    transcript_files=[file.filename for file in fs_bucket.find({"filename": {"$regex": r"\.txt$"}})]
    return render_template("index.html",transcript_files=transcript_files)

@app.route("/download/<filename>")
def download_file(filename):
    file_cursor = fs_bucket.find({"filename": filename})
    file = next(file_cursor, None)
    if not file:
        abort(404)
    
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = "application/octet-stream"

    return Response(
        file.read(),
        mimetype=mime_type,
        headers={"Content-Disposition":f"attachment; filename={filename}"}
    )
    #return redirect(url_for("home"))

@app.route("/view/<filename>")
def view_file(filename):
    #finding filename
    file_cursor = fs_bucket.find({"filename": filename})
    file = next(file_cursor, None)
    if not file:
        return abort(404)
    
    if filename.endswith(".txt"):
        content=file.read().decode("utf-8")
        return Response(content, mimetype="text/plain")
    mime_type, _ = mimetypes.guess_type(filename)
    return Response(file.read(), mimetype=mime_type or "application/octet-stream")
    #return redirect(url_for("home"))

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chatbot/query", methods=["POST"])
def chatbot_query():
    try:
        session_id=request.json.get("session_id")
        print(session_id)
        role=request.json.get("role")
        user_input = request.json.get("user_input")
        print(f"role is {role}")
        if not user_input:
            return jsonify({"error": "Invalid input"}), 400
        
        previous_answer=[]
        previous_question=""
        previous_role=""
        if session_id in conversations:
            previous_question=conversations[session_id].get("previous_question","")
            previous_answer=conversations[session_id].get("previous_answer",[])
            previous_role=conversations[session_id].get("previous_role","")


        # Process user query through DissertationQueryProcessor
        retrieved_chunks = query_processor.qdrant_retrieve(user_input)
        input_state = {
        "retreived_chunks": retrieved_chunks,
        "role": role,
        "model": "mistral",
        "previous_role":previous_role,
        "previous_question":previous_question,
        "previous_answer":previous_answer,
        "current_question":user_input
    }
        response=asyncio.run(run_chatbot(input_state))

        #updating session memory
        conversations[session_id]={
            "previous_question":user_input,
            "previous_answer":response["rephrased_chunks"],
            "previous_role":role

        }
        return jsonify({"response": response["rephrased_chunks"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def clear_files():
    directories=["uploads"]
    set=False
    for directory in directories:
        try:
            set=True
            if not os.path.exists(directory):
                print(f"Directory {directory} path is incorrect or does not exist")
                set=False
            if not os.path.isdir(directory):
                print(f"Directory {directory} is not a directory")
                set=False
            for file in os.listdir(directory):
                print(file)
                file_path=os.path.join(directory,file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"Skipping {file_path} for deletion")
            set=True
        except Exception as e:
            print(f"An error occurred: {e}")
            set= False
    if set==True:
        return True
    else:
        return False

@app.route("/clear")
def delete_all_files():
    success=clear_files()
    if success:
        return redirect(url_for("home"))
    else:
        return "Failed to delete Files",500
    
@app.route('/chatbot/voice', methods=['POST'])
def voice_input():
    try:
        # Use the microphone as input
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("Listening for voice input...")
            audio2 = r.listen(source2)

            # Convert speech to text
            MyText = r.recognize_google(audio2).lower()
            print(f"Recognized: {MyText}")

            if MyText == "exit":
                return jsonify({'transcription': '', 'message': 'Exit command received'})

            return jsonify({'transcription': MyText})

    except sr.RequestError:
        return jsonify({'error': 'Speech Recognition API unavailable'}), 500
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

