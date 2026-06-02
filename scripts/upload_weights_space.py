#!/usr/bin/env python3
"""
Hugging Face Space Python Uploader

This script uploads the local packaged space ('weights_space/') directly to your 
Hugging Face Gradio Space using the official Python API. It completely bypasses 
the need for a local Git or Git LFS installation!

Usage:
    python3 scripts/upload_weights_space.py
"""

import os
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SPACE_DIR = ROOT_DIR / "weights_space"

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
    print("🚀 Hugging Face Space Python Uploader 🚀")
    print("=" * 60)

    if not check_huggingface_hub():
        return

    from huggingface_hub import HfApi, login

    # Ensure packaged space exists
    if not SPACE_DIR.exists() or not any(SPACE_DIR.iterdir()):
        print(f"❌ Error: packaged space folder not found at: {SPACE_DIR}")
        print("💡 Please run 'python3 scripts/deploy_weights_space.py' first.")
        return

    # 1. Login to Hugging Face
    print("\n🔑 Step 1: Log in to Hugging Face")
    print("---------------------------------")
    print("👉 Go to: https://huggingface.co/settings/tokens")
    print("👉 Create a 'WRITE' token, copy it, and paste it below:")
    
    token = input("HF Access Token (Write): ").strip()
    if token:
        try:
            login(token=token)
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return
    else:
        print("ℹ️ Attempting to use existing Hugging Face session...")

    # Get API handle
    api = HfApi()
    
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Authenticated successfully as user: {username}")
    except Exception as e:
        print("❌ Authentication failed. A valid WRITE token is required.")
        return

    # 2. Configure target space
    print("\n📦 Step 2: Target Space Configuration")
    print("-------------------------------------")
    space_name = input("Enter your Hugging Face Space name (default: weights): ").strip()
    if not space_name:
        space_name = "weights"

    space_id = f"{username}/{space_name}"
    print(f"🎯 Target Space: https://huggingface.co/spaces/{space_id}")

    # 3. Direct folder upload
    print(f"\n📤 Step 3: Uploading Space files to {space_id}...")
    print("--------------------------------------------------")
    print("⏳ Starting upload. Large files will display individual progress bars.")
    print("💡 This bypasses Git LFS and uses direct high-speed HTTP streams!")

    try:
        api.upload_folder(
            folder_path=str(SPACE_DIR),
            repo_id=space_id,
            repo_type="space"
        )
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your Gradio Weights Space is completely deployed and running!")
        print("=" * 60)
        print(f"🔗 Space URL: https://huggingface.co/spaces/{space_id}")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
