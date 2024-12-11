for folder in */; do
  cd "$folder"
  for file in *.avi; do
    ffmpeg -i "$file" -vcodec libx264 -crf 28 "compressed_$file"
  done
  cd ..
done