from fastapi import FastAPI, UploadFile, File
from rag import ask_question, process_pdf
import os
import shutil

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):

    for old_file in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, old_file))

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_pdf(file_path)

    return {
        "message": "PDF uploaded and indexed successfully"
    }

@app.get("/ask/")
async def ask(query: str):

    answer = ask_question(query)

    return {"query": query, "answer": answer}