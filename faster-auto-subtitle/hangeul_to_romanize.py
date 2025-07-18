# pip install hangul-romanize

from hangul_romanize import Transliter
from hangul_romanize.rule import academic

transliter = Transliter(academic)

def romanize_line(text):
    try:
        return transliter.translit(text)
    except Exception:
        return text  # fallback for non-Korean lines or errors

def process_srt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            if any('\uAC00' <= char <= '\uD7A3' for char in line):  # Hangul check
                f.write(line)
                f.write(romanize_line(line.strip()) + '\n')  # Add romanization as next line
            else:
                f.write(line)

# Example usage
# process_srt("korean_original.srt", "korean_romanized.srt")
process_srt("BerryGood - dont believe MV - korean.srt", "BerryGood - dont believe MV - romanize.srt")
