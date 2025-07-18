# for ep in {30..35}
for ep in {02..20}
do
  # file="/home/muhammad/Downloads/videos3/the_saga_of_the_lost_kingdom/the_saga_of_the_lost_kingdom_ep_${ep}.mp4"
  file="/home/muhammad/Downloads/videos3/the_maverick_1982/the_maverick_1982_ep_${ep}.mp4"
  if [[ -f "$file" ]]; then
    echo "Processing $file..."
    faster_auto_subtitle --language zh --target_language en --task translate "$file"
  else
    echo "File $file not found, skipping..."
  fi
done

