import re
import tempfile
from pathlib import Path
from tqdm import tqdm
import ffmpeg
from faster_whisper import WhisperModel

# Optional transliteration/translation tools
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
import pykakasi
from transformers import MarianMTModel, MarianTokenizer

# Setup transliterators
kakasi = pykakasi.kakasi()
korean_transliter = Transliter(academic)

# Supported languages for enhancement
SUPPORTED_LANGS = {
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese"
}

# Load translation model for Korean
ko_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
ko_translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")


def format_timestamp(seconds: float) -> str:
    millisec = int(seconds * 1000)
    h = millisec // 3600000
    m = (millisec % 3600000) // 60000
    s = (millisec % 60000) // 1000
    ms = millisec % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def whisper_transcribe(video_path: str, model_size="large-v3", language="auto"):
    # Extract audio to WAV
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True, quiet=True)

    # Transcribe
    # model = WhisperModel(model_size, compute_type="auto")
    model = WhisperModel(model_size, compute_type="auto", local_files_only=False)
    segments, info = model.transcribe(audio_path, beam_size=5, language=None if language == "auto" else language)

    detected_lang = info.language
    return list(segments), detected_lang


def translate_korean(text):
    inputs = ko_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = ko_translator.generate(**inputs)
    return ko_tokenizer.decode(translated[0], skip_special_tokens=True)


def romanize_line(text: str, lang: str) -> tuple[str, str]:
    if lang == "ko":
        roman = korean_transliter.translit(text)
        translation = translate_korean(text)
        return roman, translation

    elif lang == "ja":
        result = kakasi.convert(text)
        roman = " ".join([item["hepburn"] for item in result])
        return roman, ""

    elif lang == "zh":
        # Placeholder: You can add Chinese romanization (e.g. pypinyin) here
        return "", ""

    return "", ""


def enhance_segments_to_srt(segments, output_path: str, lang: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(tqdm(segments)):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = seg.text.strip()

            lines = [text]

            if lang in SUPPORTED_LANGS:
                roman, translation = romanize_line(text, lang)
                if roman:
                    lines.append(roman)
                if translation:
                    lines.append(translation)

            block = f"{i+1}\n{start} --> {end}\n" + "\n".join(lines) + "\n\n"
            f.write(block)


def generate_subtitles(video_path: str, output_srt: str, model_size="large-v3", language="auto") -> str:
    print(f"📂 Processing: {video_path}")
    print(f"🔎 Transcribing using Whisper model: {model_size} (Language: {language})")

    segments, detected_lang = whisper_transcribe(video_path, model_size, language)

    print(f"📝 Enhancing subtitles with romanization and English for language: {detected_lang}")
    if detected_lang not in SUPPORTED_LANGS:
        print(f"⚠️ Language '{detected_lang}' is not supported for enhancement. Saving raw subtitles.")
    else:
        print(f"✅ Detected language: {SUPPORTED_LANGS[detected_lang]}")

    enhance_segments_to_srt(segments, output_srt, detected_lang)

    print(f"✅ Subtitle saved to: {output_srt}")
    return output_srt
