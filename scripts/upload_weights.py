#!/usr/bin/env python3
"""
Hugging Face Model Weights Uploader

This script guides you through uploading your local model weights (SAM, Grounding DINO, 
and cached BERT models) to a dedicated Model Repository on Hugging Face.

Usage:
    python scripts/upload_weights.py
"""

import os
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"

def check_huggingface_hub():
    """Ensure huggingface_hub is installed"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("⚠️ 'huggingface_hub' package is required.")
        print("💡 Installing it now...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            print("✅ 'huggingface_hub' successfully installed!")
            return True
        except Exception as e:
            print(f"❌ Failed to install huggingface_hub: {e}")
            print("👉 Run manually: pip install huggingface_hub")
            return False

def main():
    print("=" * 60)
    print("🤖 Labelground Hugging Face Weights Uploader 🤖")
    print("=" * 60)

    if not check_huggingface_hub():
        return

    from huggingface_hub import HfApi, create_repo, login

    # Ensure weights folder exists and has files
    if not WEIGHTS_DIR.exists() or not any(WEIGHTS_DIR.iterdir()):
        print(f"❌ Weights directory not found or empty at: {WEIGHTS_DIR}")
        print("💡 Please run 'python scripts/setup_weights.py' first to download your models.")
        return

    print("\n📂 Scanning local weights folder...")
    for item in WEIGHTS_DIR.iterdir():
        if item.is_file():
            print(f"  • {item.name} ({item.stat().st_size / (1024*1024):.1f} MB)")
        elif item.is_dir():
            dir_size = sum(f.stat().st_size for f in item.glob('**/*') if f.is_file())
            print(f"  • {item.name}/ ({dir_size / (1024*1024):.1f} MB)")

    # 1. Login to Hugging Face
    print("\n🔑 Step 1: Log in to Hugging Face")
    print("---------------------------------")
    print("👉 If you are already logged in via CLI, you can press Enter.")
    print("👉 Otherwise, go to https://huggingface.co/settings/tokens")
    print("👉 Create a 'WRITE' token, copy it, and enter it below:")
    
    token = input("HF Access Token (Write): ").strip()
    if token:
        try:
            login(token=token)
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return
    else:
        print("ℹ️ Using existing Hugging Face credentials.")

    # Get API handle
    api = HfApi()
    
    # Try to resolve username
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Authenticated successfully as user: {username}")
    except Exception as e:
        print("❌ Could not verify Hugging Face login session.")
        print("👉 Please run 'huggingface-cli login' in your terminal, then rerun this script.")
        return

    # 2. Get Repository Name
    print("\n📦 Step 2: Configure Repository")
    print("-------------------------------")
    repo_name = input("Enter a name for your HF Model Repository (default: labelground-weights): ").strip()
    if not repo_name:
        repo_name = "labelground-weights"

    repo_id = f"{username}/{repo_name}"
    
    is_private_input = input("Make this model repository private? (y/n, default: n): ").strip().lower()
    private = is_private_input == 'y'

    # 3. Create the Model Repository
    print(f"\n🚀 Creating Model Repository: {repo_id}...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository is active at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    # 4. Upload weights folder
    print(f"\n📤 Step 3: Uploading weights to {repo_id}...")
    print("--------------------------------------------------")
    print("⏳ Starting upload. Large files will show detailed progress bars.")
    
    try:
        api.upload_folder(
            folder_path=str(WEIGHTS_DIR),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[".cache*", "cache*", "matplotlib*", "__pycache__*"]
        )
        print("\n🎉 SUCCESS! All model weights are uploaded and hosted on Hugging Face!")
        print("=" * 60)
        print(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
        print("\n👉 To download these weights in the future, anyone can run:")
        print(f"   huggingface-cli download {repo_id} --local-dir weights")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error during folder upload: {e}")

if __name__ == "__main__":
    main()
