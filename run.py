from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import uvicorn
import whisper
import asyncio
import os
import subprocess
import torch

app = FastAPI()

@app.on_event("startup")
async def startup():
    MODEL_DIR = os.getenv("ASR_MODEL_PATH", "/home/models/whisper")
    MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(MODEL_DIR):
            print(f"📁 Création du dossier des modèles : {MODEL_DIR}")
            os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"⏳ Chargement du modèle {MODEL_NAME}...")
    try:
        app.state.model = whisper.load_model(MODEL_NAME, device=DEVICE, download_root=MODEL_DIR)
    except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")

    print(f"✅ Modèle {MODEL_NAME} prêt sur {DEVICE}")
    app.state.is_processing = False

async def convert_to_wav(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, output_path]
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    await process.communicate()

# Reçoit un fichier audio et retourne la transcription
@app.post("/BatchTranscriptionService")
async def transcribe_audio(audioFile: UploadFile = File(...)):
    # Vérifie si une transcription est déjà en cours
    if app.state.is_processing:
        raise HTTPException(409, "Service occupé : transcription en cours")

    app.state.is_processing = True

    temp_path = "temp"
    with open(temp_path, "wb") as buffer:
        buffer.write(await audioFile.read())

    wav_path = f"{temp_path}.wav"
    await convert_to_wav(temp_path, wav_path)

    try:
        result = app.state.model.transcribe(wav_path, language="fr")
        # await asyncio.sleep(10) # simule une transcription lente
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        app.state.is_processing = False

    filtered_segments = [
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ]

    return {
            "full_text": result["text"],
            "segments": filtered_segments,
            "language": result.get("language")
        }

@app.get("/busy")
async def is_busy():
    return {"is_processing": app.state.is_processing}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)