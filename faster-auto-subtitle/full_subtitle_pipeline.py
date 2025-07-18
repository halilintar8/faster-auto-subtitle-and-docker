# pip install -r requirements.txt
# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# Run this (no need to compile from source) for gpu nvidia rtx 4060 :
# pip install --upgrade pip
# pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
#   --extra-index-url https://download.pytorch.org/whl/cu121


import subprocess
import os
import sys
import re
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from transformers import MarianMTModel, MarianTokenizer

# --------- Configuration ---------
MODEL_NAME = "Helsinki-NLP/opus-mt-ko-en"
LANG = "ko"
TASK = "transcribe"
WHISPER_MODEL = "large-v3"
OUTPUT_TYPE = "srt"
# ----------------------------------

# Initialize transliteration and translation
transliter = Transliter(academic)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def romanize_line(text):
    return transliter.translit(text)

def translate_line(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def process_srt_bilingual(input_srt, output_srt):
    with open(input_srt, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    new_blocks = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            new_blocks.append(block)
            continue

        index = lines[0]
        timestamp = lines[1]
        text_lines = lines[2:]

        new_text_lines = []
        for line in text_lines:
            if any('\uAC00' <= char <= '\uD7A3' for char in line):  # Hangul detected
                roman = romanize_line(line)
                english = translate_line(line)
                new_text_lines.extend([line, roman, english])
            else:
                new_text_lines.append(line)

        new_block = f"{index}\n{timestamp}\n" + "\n".join(new_text_lines)
        new_blocks.append(new_block)

    with open(output_srt, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(new_blocks))

def run_whisper(video_path, output_srt):
    cmd = [
        "faster_auto_subtitle",
        "--language", LANG,
        "--task", TASK,
        "--model", WHISPER_MODEL,
        "--output_type", OUTPUT_TYPE,
        video_path
    ]
    print("Running Whisper transcription...")
    subprocess.run(cmd, check=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    srt_file = f"{base_name}.srt"
    return srt_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python full_subtitle_pipeline.py /path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Processing: {video_path}")
    srt_file = run_whisper(video_path, output_srt=None)

    output_srt = srt_file.replace(".srt", " - bilingual.srt")
    print("Enhancing subtitle with Romanized + English...")
    process_srt_bilingual(srt_file, output_srt)
    print(f"✅ Done: {output_srt}")

if __name__ == "__main__":
    main()

