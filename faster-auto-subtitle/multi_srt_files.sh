#!/bin/bash

BASE_DIR="/home/muhammad/Downloads/videos3/dream_paradise"

for i in $(seq -w 1 10); do
  FILE="$BASE_DIR/dream_paradise_ep_${i}.mp4"
  if [[ -f "$FILE" ]]; then
    echo "✅ Processing: $FILE"
    faster_auto_subtitle --target_language en --task translate --output_type srt "$FILE"
  else
    echo "❌ File not found: $FILE"
  fi
done
