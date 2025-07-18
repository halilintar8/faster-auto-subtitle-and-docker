# api_main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import tempfile
import shutil
from pathlib import Path

from subtitle_pipeline import generate_subtitles

app = FastAPI()

@app.post("/generate-subtitles/")
async def generate_subtitles_api(
    model: str = Form("large-v3"),
    language: str = Form("auto"),
    file: UploadFile = File(...)
):
    # Save uploaded file to temp location
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    with open(temp_video.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define subtitle output path
    output_srt = Path(temp_video.name).with_suffix(".srt")

    # Run transcription + enhancement pipeline
    # generate_subtitles(str(temp_video.name), str(output_srt), model=model, language=language)
    generate_subtitles(str(temp_video.name), str(output_srt), model_size=model, language=language)

    return FileResponse(
        str(output_srt), 
        filename=output_srt.name, 
        media_type="application/x-subrip"
    )
