import argparse
import re
from pathlib import Path
import tempfile

from faster_whisper import WhisperModel
from tqdm import tqdm
import ffmpeg

# Korean Romanizer
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

# Japanese Romanizer
import pykakasi

# Chinese Pinyin
from pypinyin import lazy_pinyin

# Translation model
from transformers import MarianMTModel, MarianTokenizer

SUPPORTED_LANGUAGES = {
    "ko": {
        "romanizer": lambda text: Transliter(academic).translit(text),
        "translator_model": "Helsinki-NLP/opus-mt-ko-en"
    },
    "ja": {
        "romanizer": lambda text: " ".join([item['hepburn'] for item in pykakasi.kakasi().convert(text)]),
        "translator_model": "Helsinki-NLP/opus-mt-ja-en"
    },
    "zh": {
        "romanizer": lambda text: " ".join(lazy_pinyin(text)),
        "translator_model": "Helsinki-NLP/opus-mt-zh-en"
    }
}


def whisper_transcribe(video_path, output_srt, model_size="large-v3", language="auto"):
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True, quiet=True)

    model = WhisperModel(model_size, compute_type="auto")
    segments, info = model.transcribe(audio_path, beam_size=5, language=None if language == "auto" else language)

    detected_language = info.language if language == "auto" else language

    def format_timestamp(seconds: float):
        millisec = int(seconds * 1000)
        h = millisec // 3600000
        m = (millisec % 3600000) // 60000
        s = (millisec % 60000) // 1000
        ms = millisec % 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(tqdm(segments)):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"{i+1}\n{start} --> {end}\n{segment.text.strip()}\n\n")

    return detected_language


def load_translation_model(lang_code):
    model_name = SUPPORTED_LANGUAGES[lang_code]["translator_model"]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def enhance_srt(input_file, output_file, lang_code):
    if lang_code not in SUPPORTED_LANGUAGES:
        print(f"⚠️ Language '{lang_code}' is not supported for romanization/translation. Skipping enhancement.")
        return

    romanizer = SUPPORTED_LANGUAGES[lang_code]["romanizer"]
    tokenizer, translator = load_translation_model(lang_code)

    with open(input_file, 'r', encoding='utf-8') as f:
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
            try:
                roman = romanizer(line)
                translated = translate(line, tokenizer, translator)
                new_text_lines.extend([line, roman, translated])
            except Exception:
                new_text_lines.append(line)

        new_block = f"{index}\n{timestamp}\n" + "\n".join(new_text_lines)
        new_blocks.append(new_block)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(new_blocks))


def main():
    parser = argparse.ArgumentParser(description="🎞️ Auto Subtitler with Romanization + Translation")
    parser.add_argument("input_video", help="Path to input video file (e.g. .mp4)")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (e.g. tiny, base, small, large-v3)")
    parser.add_argument("--language", default="auto", help="Language code (e.g. ko, ja, zh) or 'auto'")
    parser.add_argument("--output", default=None, help="Optional custom .srt output filename")
    args = parser.parse_args()

    video_path = Path(args.input_video)
    output_srt = Path(args.output) if args.output else video_path.with_suffix(".srt")

    print(f"📂 Processing: {video_path}")
    print(f"🔎 Transcribing using Whisper model: {args.model} (Language: {args.language})")
    detected_lang = whisper_transcribe(str(video_path), str(output_srt), model_size=args.model, language=args.language)

    print(f"📝 Enhancing subtitles with romanization and English for language: {detected_lang}")
    enhance_srt(str(output_srt), str(output_srt), detected_lang)

    print(f"✅ Subtitle saved to: {output_srt}")


if __name__ == "__main__":
    main()



# requirements :

# pip install \
#   faster-whisper==1.0.3 \
#   ffmpeg-python==0.2.0 \
#   tqdm==4.66.4 \
#   transformers>=4.4,<5 \
#   numpy>=1.24.2,<=1.26.4 \
#   nltk<=3.8.1 \
#   sentencepiece<=0.2.0 \
#   protobuf<=5.27.2 \
#   huggingface_hub<=0.23.5 \
#   hangul-romanize \
#   torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 \
#   sacremoses


# how to install :

# pip install -r requirements.txt

# pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
#   --extra-index-url https://download.pytorch.org/whl/cu121

# Visit https://pytorch.org/get-started/locally/
# Select your CUDA version to get the correct install command


# usage :

# python full_auto_subtitle_pipeline.py "/videos/song.mp4"

# python full_auto_subtitle_pipeline.py \
#   --model large-v3 \
#   --language auto \
#   --output final.srt \
#   "/videos/song.mp4"

# python full_auto_subtitle_pipeline.py \
#   --model large-v3 \
#   --language auto \
#   "/home/muhammad/Downloads/videos3/test_subtitles/BerryGood - dont believe MV.mp4"

# python full_auto_subtitle_pipeline.py --model large-v3 --language auto "/home/muhammad/Downloads/videos3/test_subtitles/ONE_OK_ROCK-Wherever_you_are.mp4"


# example :

# (autosub-env) 󰣇 latihan_bintang5/faster_autosub/faster-auto-subtitle   main  !? ❯ python full_auto_subtitle_pipeline.py \                                                                  autosub-env 3.12.3  17:03 
#   --model large-v3 \
#   --language auto \
#   "/home/muhammad/Downloads/videos3/test_subtitles/BerryGood - dont believe MV.mp4"

# 📂 Processing: /home/muhammad/Downloads/videos3/test_subtitles/BerryGood - dont believe MV.mp4
# 🔎 Transcribing using Whisper model: large-v3 (Language: auto)
# 81it [00:30,  2.67it/s]
# 📝 Enhancing subtitles with romanization and English...
# ✅ Subtitle saved to: /home/muhammad/Downloads/videos3/test_subtitles/BerryGood - dont believe MV.srt
# (autosub-env) 󰣇 latihan_bintang5/faster_autosub/faster-auto-subtitle   main  !? ❯                             


