---
title: Labelground
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Labelground

**A self-hosted, offline-first annotation platform powered by an AI ensemble.**

Labelground runs entirely on your machine. It takes a video or a folder of images, lets an AI take the first pass at annotating them, and then lets you correct, verify, and export the result in formats like YOLO, COCO, and Pascal VOC. The more you correct, the smarter it gets — it retrains itself in the background using your verified labels.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg?logo=docker)](https://www.docker.com/)

---

## What it does

- **Auto-annotation** — Drop in your images or a video file and let the AI draw bounding boxes and polygon masks automatically. No prompting needed, but you can guide it with text if you want specific classes.
- **Three models working together** — YOLO-World handles fast initial detection, Grounding DINO refines it with text-guided reasoning, and SAM draws the precise pixel-level boundary.
- **Active learning loop** — Every time you verify a frame, it counts. Once enough frames are verified, the platform kicks off a fine-tuning run in the background so future predictions on your data get better.
- **Completely offline** — Nothing leaves your machine. No API keys, no cloud uploads, no tracking.
- **Export anywhere** — When you're done, export your dataset in YOLO, COCO JSON, or Pascal VOC format, split into train/val/test sets.

---

## Project layout

```
labelground/
├── backend/
│   ├── ai_ensemble.py        # coordinates YOLO-World, Grounding DINO, and SAM
│   ├── ai_service.py         # lower-level model loading and inference
│   ├── augmentation.py       # flips, blur, brightness shifts — 10x dataset expansion
│   ├── auth.py               # JWT login, bcrypt password hashing
│   ├── export.py             # YOLO / COCO / VOC export logic
│   ├── main.py               # all API routes
│   ├── training_orchestrator.py  # background thread watching for training triggers
│   └── upload_endpoints.py   # handles video uploads and large file streaming
├── database/
│   └── models.py             # SQLAlchemy models for users, projects, annotations
├── filesystem/
│   └── workspace.py          # per-project folder management, ffmpeg frame extraction
├── scripts/
│   ├── setup_weights.py      # downloads all model weights automatically
│   ├── inspect_db.py         # prints a summary of your database contents
│   ├── migrate_db.py         # runs safe schema migrations
│   └── upload_weights.py     # (maintainer tool) pushes weights to Hugging Face
├── static/                   # the frontend (HTML, CSS, JS — no framework)
├── legacy/                   # old drafts and experiments, kept for reference
├── Dockerfile                # builds the full CUDA environment from scratch
├── build.sh                  # runs docker build
├── run.sh                    # runs the container
├── start.sh                  # startup script (used inside the container)
├── run.py                    # local entrypoint without Docker
├── config.yaml               # global settings
└── requirements.txt          # Python dependencies for local setup
```

---

## Getting started

There are two ways to run this: with Docker (recommended, handles all dependencies) or directly on your machine (faster to start, but you need CUDA set up yourself).

### Option 1 — Docker (recommended)

This is the easiest path. Docker handles the CUDA environment, all system libraries, and Python packages for you.

**What you need before starting:**
- [Docker](https://docs.docker.com/get-docker/) installed
- An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) set up
- About 20 GB of free disk space for the image

**Step 1 — Clone the repository**

```bash
git clone https://github.com/thamizj2005/labelground.git
cd labelground
```

**Step 2 — Download the model weights**

The AI models are hosted on Hugging Face and need to be pulled before you build the image.

```bash
pip install huggingface_hub
huggingface-cli download thamizhg/labelground-weights --local-dir weights
```

This downloads roughly 4 GB. Go make a coffee.

**Step 3 — Build the Docker image**

```bash
chmod +x build.sh
./build.sh
```

This compiles GroundingDINO's CUDA operators and installs everything inside the container. First build takes 10–20 minutes depending on your connection and machine. Subsequent builds are cached and much faster.

**Step 4 — Run it**

```bash
chmod +x run.sh
./run.sh
```

Open your browser and go to `http://localhost:8000`.  
Default login: **admin / admin123**

---

### Option 2 — Local Python setup (no Docker)

If you already have a working CUDA environment and don't want the overhead of Docker, you can run it directly.

**What you need:**
- Python 3.10 or newer
- CUDA toolkit matching your PyTorch version (PyTorch 2.0+ recommended)
- `ffmpeg` installed system-wide (for video frame extraction)

**Step 1 — Clone and set up a virtual environment**

```bash
git clone https://github.com/thamizj2005/labelground.git
cd labelground

python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Step 2 — Download the model weights**

```bash
python scripts/setup_weights.py
```

This script downloads SAM, Grounding DINO, and the BERT tokenizer (about 4–5 GB total) and puts them in the `weights/` folder.

Alternatively, pull from Hugging Face directly:

```bash
pip install huggingface_hub
huggingface-cli download thamizhg/labelground-weights --local-dir weights
```

**Step 3 — Start the server**

```bash
chmod +x start.sh
./start.sh
```

Or just run Python directly:

```bash
python run.py
```

Open `http://localhost:8000` in your browser.  
Default login: **admin / admin123**

---

## How annotation works

Once you're in, the workflow goes like this:

1. **Create a project** and upload either images or an `.mp4` file. Videos get split into frames automatically.
2. **Run auto-annotation** on a frame or a batch. The backend sends the image through YOLO-World and Grounding DINO to get bounding boxes, then passes those to SAM to get polygon masks.
3. **Review and correct** the results on the canvas. Drag vertices, redraw polygons, delete wrong labels, add new ones.
4. **Mark frames as verified.** Once you hit the verification threshold (configurable in `config.yaml`), the training orchestrator wakes up and fine-tunes the model on your verified frames in the background.
5. **Export** when ready. Choose your format (YOLO / COCO / VOC) and your split ratios (e.g., 80/10/10 train/val/test).

---

## Model weights

All model weights are hosted for free on Hugging Face:

👉 [https://huggingface.co/thamizhg/labelground-weights](https://huggingface.co/thamizhg/labelground-weights)

| Model | File | Size |
|---|---|---|
| SAM ViT-H | `sam_vit_h_4b8939.pth` | 2.56 GB |
| SAM ViT-B | `sam_vit_b_01ec64.pth` | 375 MB |
| Grounding DINO | `groundingdino_swint_ogc.pth` | 694 MB |
| YOLO-World | `yolov8l-worldv2.pt` | 94 MB |
| BERT tokenizer | `bert-base-uncased/` | ~420 MB |

Download them all with one command:

```bash
huggingface-cli download thamizhg/labelground-weights --local-dir weights
```

---

## Tech stack

- **FastAPI + Uvicorn** — async REST backend
- **SQLAlchemy + SQLite (WAL mode)** — metadata and annotation storage
- **PyTorch + CUDA** — model inference and fine-tuning
- **Ultralytics YOLO-World** — open-vocabulary object detection
- **IDEA-Research Grounding DINO** — text-guided zero-shot detection
- **Meta Segment Anything (SAM)** — polygon boundary extraction
- **OpenCV** — frame processing, augmentations
- **bcrypt + python-jose** — authentication

---

## License

MIT. Do whatever you want with it, just don't hold me liable.
