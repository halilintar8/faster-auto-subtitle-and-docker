# pip install transformers torch hangul-romanize
# pip install torch==2.3.0 torchvision==0.18.0
# pip install torchaudio==2.3.0 


from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from transformers import MarianMTModel, MarianTokenizer
import re

# Initialize transliterator and translation model
transliter = Transliter(academic)

# Load Opus-MT Korean to English model (offline capable if cached)
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def romanize_line(text):
    return transliter.translit(text)

def translate_line(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def process_srt_bilingual(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split blocks
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

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(new_blocks))

# Example usage
process_srt_bilingual(
    # "BerryGood - dont believe MV - korean.srt",
    "BerryGood - dont believe MV.srt",
    "BerryGood - dont believe MV - bilingual.srt"
)

