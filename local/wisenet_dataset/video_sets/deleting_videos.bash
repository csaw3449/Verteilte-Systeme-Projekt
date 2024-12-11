#!/bin/bash
for folder in */; do
  cd "$folder"
  for file in *; do
  if [[ -f "$file" && "$file" != compressed* ]]; then
    rm -- "$file"
  fi
done
  cd ..
done