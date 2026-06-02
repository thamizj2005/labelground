#!/usr/bin/env python3
"""
Hugging Face Weights Space Deployer (Strict Clean Packager)

This script automates setting up the dedicated Gradio Space workspace ('weights_space/')
to host your 5 GB model weights and exposes a beautiful web download dashboard.
It performs a strict cleanup to ignore any temporary local caches (like .cache/ or matplotlib/).

Usage:
    python scripts/deploy_weights_space.py
"""

import os
import shutil
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
SPACE_DIR = ROOT_DIR / "weights_space"

# Gradio Dashboard Code for app.py
GRADIO_APP_CODE = """import gradio as gr
import os
from pathlib import Path

# Helper to format sizes
def get_file_size(path):
    if not path.exists():
        return "Not Loaded"
    size = path.stat().st_size / (1024 * 1024)
    return f"{size:.1f} MB"

# Build beautiful interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🤖 Labelground Model Weights Hub")
    gr.Markdown("Welcome to the official offline model weights repository for **Labelground** — the Auto-Adaptive Annotation Platform.")
    
    with gr.Tab("📦 Model Catalog"):
        gr.Markdown("### Foundational AI Ensemble Checkpoints")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🟢 Segment Anything Model (SAM)")
                gr.Markdown("**File:** `weights/sam_vit_b_01ec64.pth`\\n**Size:** ~375 MB\\n**Role:** Interactive sub-pixel semantic polygon boundary extraction.")
            
            with gr.Column():
                gr.Markdown("#### 🔵 Grounding DINO")
                gr.Markdown("**File:** `weights/groundingdino_swint_ogc.pth`\\n**Size:** ~694 MB\\n**Role:** Text-prompt guided open-vocabulary bounding box hypothesis.")

        gr.Markdown("---")
        gr.Markdown("#### 🎓 Text Encoder (BERT Base Uncased)")
        gr.Markdown("**Directory:** `weights/bert-base-uncased/`\\n**Size:** ~400 MB\\n**Role:** High-fidelity token embeddings for complex prompt resolution.")

    with gr.Tab("🚀 Quick Download (CLI)"):
        gr.Markdown("### Download and Mount Directly into Labelground")
        gr.Markdown("You can pull these weights completely offline into your local environment with one command:")
        gr.Code(
            code="pip install huggingface_hub\\nhuggingface-cli download thamizhg/weights --local-dir weights --repo-type space",
            language="bash"
        )
        gr.Markdown("💡 *Simply run this inside your local 'vision/' root directory to sync the entire offline catalog!*")

    gr.Markdown("\\n---")
    gr.Markdown("🔒 *Labelground is completely offline-first. This public Hugging Face Space is only used to host and distribute the heavy foundational neural weights at $0 cost.*")

# Run app
if __name__ == "__main__":
    demo.launch()
"""

# Space README Metadata Frontmatter
SPACE_README_CONTENT = """---
title: weights
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# Labelground Weights Hub
This Space hosts the offline models and segmenting parameters for Labelground.
"""

def main():
    print("=" * 60)
    print("🚀 Labelground Hugging Face Space Packager 🚀")
    print("=" * 60)

    # 1. Verify weights exist
    if not WEIGHTS_DIR.exists() or not any(WEIGHTS_DIR.iterdir()):
        print(f"❌ Error: Weights directory is empty at: {WEIGHTS_DIR}")
        print("💡 Run 'python scripts/setup_weights.py' to download the weights first!")
        return

    # 2. Re-create clean space folder (wiping out old corrupt states)
    if SPACE_DIR.exists():
        print(f"🧹 Cleaning up existing workspace at: {SPACE_DIR.relative_to(ROOT_DIR)}...")
        shutil.rmtree(SPACE_DIR)
        
    print(f"📂 Creating clean Space folder at: {SPACE_DIR.relative_to(ROOT_DIR)}...")
    SPACE_DIR.mkdir(parents=True, exist_ok=True)
    (SPACE_DIR / "weights").mkdir(parents=True, exist_ok=True)

    # 3. Write app.py and README.md
    print("🛠️ Writing Gradio app dashboard...")
    with open(SPACE_DIR / "app.py", "w") as f:
        f.write(GRADIO_APP_CODE)

    print("📄 Writing Space metadata...")
    with open(SPACE_DIR / "README.md", "w") as f:
        f.write(SPACE_README_CONTENT)

    # 4. Copy ONLY valid model weights (skipping dotfiles and caches)
    print("\n📦 Copying model weights to Space workspace...")
    
    # Define exact folders/files to allow (skip .cache, matplotlib, cache, etc.)
    allowed_extensions = {'.pth', '.pt', '.py', '.pth', '.bin', '.json', '.txt'}
    
    for item in WEIGHTS_DIR.iterdir():
        # Ignore any hidden files/folders (starting with dot) or local caches
        if item.name.startswith('.') or item.name in ('cache', 'matplotlib', '__pycache__'):
            print(f"  • Skipping temporary cache: {item.name}")
            continue
            
        dest = SPACE_DIR / "weights" / item.name
        if item.is_file():
            if item.suffix in allowed_extensions:
                print(f"  • Copying weight file: {item.name}...")
                shutil.copy2(item, dest)
            else:
                print(f"  • Skipping non-model file: {item.name}")
        elif item.is_dir():
            print(f"  • Copying model directory: {item.name}/...")
            shutil.copytree(item, dest)

    # 5. Initialize Git and Git LFS inside the space
    print("\n🛠️ Configuring local Git LFS for large files...")
    os.chdir(str(SPACE_DIR))
    
    # Run Git init inside the Space folder
    os.system("git init")
    os.system("git branch -M main")
    
    # Initialize LFS tracking
    os.system("git lfs install")
    os.system("git lfs track 'weights/*.pth'")
    os.system("git lfs track 'weights/*.pt'")
    os.system("git lfs track 'weights/*.bin'")
    
    # Copy gitattributes to be committed
    os.system("git add .gitattributes app.py README.md")

    print("\n" + "=" * 60)
    print("🎉 SUCCESS: Your local Gradio Weights Hub is fully prepared!")
    print("=" * 60)
    print("\n👉 To deploy all 5 GB of weights to your new Space, run these commands:")
    print(f"\n   1. cd weights_space")
    print(f"   2. git remote add origin https://huggingface.co/spaces/thamizhg/weights")
    print(f"   3. git add weights/")
    print(f"   4. git commit -m 'deploy: push 5GB model weights and Gradio dashboard'")
    print(f"   5. git push -u origin main --force")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
