#!/bin/bash

PROJECT_DIR=$(pwd)

echo "🚀 Starting Project from: $PROJECT_DIR"

xhost +si:localuser:$(whoami)

docker run --rm -it \
  --net=host \
  --user $(id -u):$(id -g) \
  -e DISPLAY=$DISPLAY \
  -e TZ="Asia/Kolkata" \
  -e PYTHONPATH='/app:$PYTHONPATH' \
  -e CUDA_VISIBLE_DEVICES="" \
  -e HF_HOME=/tmp/huggingface \
  -e TORCH_HOME=/tmp/torch \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PROJECT_DIR":/app \
  -v "$HOME/.cache":/home/user/.cache \
  --workdir /app \
  --entrypoint "" \
  auto_annotate_ext \
  bash
