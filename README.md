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

# Labelground: An Auto-Adaptive Offline Vision Processing and Annotation Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)

An elegant, highly automated, locally-hosted annotation and computer vision processing platform designed to accelerate dataset generation while ensuring complete data sovereignty. By employing a tripartite AI ensemble (**YOLO-World**, **Grounding DINO**, and **Segment Anything Model (SAM)**), the platform automates object detection, dynamic open-vocabulary classification, and high-fidelity polygon segmentation in a unified, completely offline environment.

---

## 🌟 Key Features

*   **Tripartite AI Ensemble:** Combines YOLO-World (for ultra-fast open-vocabulary bounding boxes), Grounding DINO (for complex, prompt-guided detection), and Meta's Segment Anything Model (SAM) for instant boundary segmentation.
*   **Active Learning Loop:** Automatically monitors manual corrections of AI-drafted labels. Once verification thresholds are met, the background orchestrator triggers custom model fine-tuning.
*   **Dynamic Data Augmentation:** Integrates OpenCV-based geometric and photometric transforms (Gaussian blur, noise, brightness shifting, flips) to expand training subsets 10x natively.
*   **Decoupled High-Performance Core:** FastAPI asynchronous ASGI architecture backed by SQLite (in WAL mode) for concurrent, responsive canvas scaling, and coordinate manipulation.
*   **Air-Gapped Privacy:** 100% offline. Zero cloud dependencies or tracking, complying with defense, medical, and enterprise data requirements.

---

## 📁 Project Structure

The project has been cleaned and organized into a professional, modular structure:

```text
vision/
├── backend/                  # FastAPI Core Backend & Services
│   ├── ai_ensemble.py        # Unified YOLO-World, Grounding DINO & SAM Service
│   ├── ai_service.py         # Standalone / Legacy AI modules
│   ├── augmentation.py       # OpenCV photometric & geometric image augmentations
│   ├── auth.py               # JWT auth & security parameters
│   ├── export.py             # YOLO, VOC, COCO dataset export managers
│   ├── main.py               # API route definitions and central endpoints
│   ├── training_orchestrator.py # Active learning background thread monitors
│   └── upload_endpoints.py   # Streaming video & large chunk file ingestion
├── database/                 # Relational Database Models
│   └── models.py             # SQLAlchemy schemas (Users, Projects, Annotations)
├── filesystem/               # Local Storage Handler
│   └── workspace.py          # FFmpeg video extraction & namespace sandboxing
├── scripts/                  # Automated Maintenance & Setup Tools
│   ├── inspect_db.py         # Database statistics inspection utility
│   ├── migrate_db.py         # Database schema update & initialization engine
│   └── setup_weights.py      # Automated model weights downloader &BERT caching
├── legacy/                   # Legacy & Prototyping Archive (Excluded from main execution)
│   ├── README_academic.md    # Original Academic Draft & Literature Review
│   ├── app.py                # Standalone video stream test script
│   ├── install_multipart.sh  # Quick installation helper script
│   └── old.py                # Previous backend server backup
├── static/                   # Frontend Web Interface
│   ├── index.html            # User authentication entry portal
│   ├── workspace.html        # Interactive multi-tool annotation canvas
│   ├── app.js                # Canvas coordinate and vertex mapper engine
│   └── styles.css            # Responsive layout & premium dark aesthetics
├── workspace/                # Local Sandboxed Storage (Git Ignored)
│   ├── meta.db               # Central SQLite metadata storage
│   └── projects/             # Active annotation projects, images & labels
├── Dockerfile                # Environment recipe to build auto_annotate_ext from scratch
├── build.sh                  # Shell script to build the local Docker environment
├── run.sh                    # Shell script to execute container with host X11 forwarding
├── run.py                    # Root Python entrypoint (FastAPI local executor)
├── start.sh                  # Automatic native startup and initialization script
├── requirements.txt          # Defined Python dependencies
└── config.yaml               # Global system configuration and default hyperparameters
```

---

## 🚀 Quick Start (Native Host Setup)

### 1. Prerequisite
Ensure you have **Python 3.10+** installed on your system.

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Clone the repository and navigate inside
cd vision

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all defined dependencies
pip install -r requirements.txt
```

### 3. Automatically Download Model Weights (5 GB)
To support the fully offline zero-shot AI ensemble, you need the pre-trained weights. We provide a single-command setup tool that writes configs, downloads the SAM/GroundingDINO weights from official repositories, and downloads/caches the local BERT tokenizer.

```bash
python scripts/setup_weights.py
```

### 4. Launch the Platform
```bash
# Make start.sh executable and run
chmod +x start.sh
./start.sh
```
Access the system interface at **`http://localhost:8000`**. The default login is `admin` / `admin123`.

---

## 🐋 Docker Environment Setup (`auto_annotate_ext`)

To run the application fully isolated with complete CUDA acceleration, you can build and execute the project using our pre-configured Docker suite:

### 1. Build the Docker Image (Free & Local)
This compiles the Grounding DINO CUDA operators and sets up all AI tools automatically inside a standard PyTorch CUDA container.
```bash
chmod +x build.sh
./build.sh
```

### 2. Run the Container
```bash
chmod +x run.sh
./run.sh
```
*Note: The container uses host network binding (`--net=host`) and X11 forwarding to allow seamless streaming, video extraction, and web API responses on port `8000`.*

---

## 💸 Cost-Free Solutions for Image & Model Weights Sharing

Distributing large Deep Learning environments (20 GB Docker image) and foundational weights (5 GB files) is notoriously expensive or bandwidth-prohibitive. Below are **100% free, industry-standard, and elegant solutions** implemented in this repository.

### 📦 1. Sharing the 20 GB Docker Environment for Free

#### Solution A: Infrastructure as Code (Recommended)
Instead of distributing a pre-compiled 20 GB binary image, you share the **`Dockerfile`** included in this repository. 
*   **Why it's perfect:** It is $0 cost, occupies 2 KB of space on Git, and is the industry-standard way to share open-source code.
*   **How others use it:** Any user clones your repo, runs `./build.sh`, and Docker compiles the exact environment dynamically on their local GPU in about 5-10 minutes.

#### Solution B: Compress & Upload to TeraBox or Google Drive
If you must share a pre-compiled container image, you can export and compress it:
1.  **Export and compress the image** (Docker files are highly repetitive; compression reduces 20 GB to ~6-8 GB):
    ```bash
    docker save auto_annotate_ext | gzip -9 > auto_annotate_ext.tar.gz
    ```
2.  **Upload for free:**
    *   **TeraBox (Recommended):** Offers **1 TB (1024 GB)** of storage completely free. Perfect for hosting heavy image tarballs.
    *   **Google Drive:** Uses the **15 GB** free tier (the compressed 7 GB tarball will fit easily).
    *   **Mega.nz:** Offers **20 GB** free cloud storage.

#### Solution C: GitHub Container Registry (GHCR)
GitHub offers **free** public container hosting under their packages service (`ghcr.io`) for public repositories.
1.  Authenticate with your GitHub Personal Access Token (PAT).
2.  Tag and push the image:
    ```bash
    docker tag auto_annotate_ext ghcr.io/your-github-username/labelground:latest
    docker push ghcr.io/your-github-username/labelground:latest
    ```
3.  Others pull it in seconds: `docker pull ghcr.io/your-github-username/labelground:latest`.

---

### 🧠 2. Sharing the 5 GB Model Weights for Free

#### Solution A: Automated Download Script (Included)
We have added a custom script **`scripts/setup_weights.py`** to the repository.
*   **Why it's perfect:** It prevents your Git repository from being bloated. It uses zero Git storage or LFS bandwidth.
*   **How it works:** When a user clones your repo and runs `python scripts/setup_weights.py`, the script fetches SAM from Meta's servers, Grounding DINO from the IDEA-Research repository, and BERT tokenizer via HuggingFace's public endpoints.

#### Solution B: Hugging Face Model Hub (100% Free)
Hugging Face is the premier, unlimited-space hosting provider for ML weights. It supports Git LFS with extremely fast download speeds at zero cost.
1.  Create a free account on [huggingface.co](https://huggingface.co/).
2.  Create a new Model Repository (e.g., `username/labelground-weights`).
3.  Upload the contents of your local `weights/` folder via the web interface or Git CLI.
4.  Provide this single download command in your README for users:
    ```bash
    pip install huggingface_hub
    huggingface-cli download username/labelground-weights --local-dir weights
    ```

#### Solution C: GitHub Releases (100% Free)
GitHub has a strict limit of 100 MB per file in Git, and Git LFS has bandwidth caps. However, **GitHub Releases are 100% free and allow assets up to 2 GB per file**.
1.  Create a release on your GitHub repository (e.g., `v1.0.0-weights`).
2.  Attach your weight files (`groundingdino_swint_ogc.pth` [694 MB] and `sam_vit_b_01ec64.pth` [375 MB]) directly to the release page.
3.  Users can download them directly from the release page or via curl:
    ```bash
    curl -L -o weights/sam_vit_b_01ec64.pth https://github.com/your-username/labelground/releases/download/v1.0.0-weights/sam_vit_b_01ec64.pth
    ```

---

## 🔄 Annotation & Active Learning Workflow

### 1. Data Ingestion
Upload static images or import an `.mp4` video. The asynchronous filesystem pipeline unrolls the video into chronological frames using optimized FFmpeg parameters based on your desired FPS.

### 2. Auto-Annotation Hypothesis
Select a single frame or batch of frames, insert the text prompts, and click **Auto-Annotate**. The backend Singleton AI Ensemble passes tensors to YOLO-World and Grounding DINO, then applies mathematical **Non-Maximum Suppression (NMS)** to resolve overlapping boundaries.

### 3. Sub-pixel Polygon Delineation
If semantic segmentation is required, the user clicks the target object boundary. The Segment Anything Model (SAM) extracts the high-resolution vector mask and outputs a serialized JSON array representing the polygon contour coordinates.

### 4. Background Active Learning
As human annotators correct AI suggestions, the database updates the frame status from `draft` to `verified`. Once the designated verification threshold (e.g., 10 images) is reached, the **`TrainingOrchestrator`** wakes up, triggers the 10x augmentation pipeline, and initiates local PyTorch fine-tuning in the background, continuously improving system predictions.

### 5. Multi-Format Export
Export the curated dataset splits (Train/Val/Test ratios) into standard formatting schemas including **YOLO text**, **Pascal VOC XML**, or **COCO JSON**.

---

## 🛠️ Tech Stack & Acknowledgments

*   **FastAPI** & **Uvicorn** for asynchronous REST backend services.
*   **SQLAlchemy** & **SQLite (WAL)** for robust metadata transaction handling.
*   **PyTorch** & **NVIDIA CUDA** for accelerated tensor computing.
*   **Ultralytics YOLOv8 & YOLO-World** for real-time open-vocabulary bounding boxes.
*   **IDEA-Research Grounding DINO** for text-guided zero-shot detection.
*   **Meta AI Segment Anything (SAM)** for interactive boundary extraction.
*   **OpenCV** & **NumPy** for sub-millisecond vision array math.
