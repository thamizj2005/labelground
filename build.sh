#!/bin/bash

echo "🛠️ Rebuilding Labelground AI Docker Image..."

# Rebuild the extended image
docker build -t auto_annotate_ext .

echo "✅ Rebuild Complete!"
echo "🚀 Now run ./run.sh to start the project."
