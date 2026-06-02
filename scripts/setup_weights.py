#!/usr/bin/env python3
"""
Model Weights Downloader & Setup Script for Labelground

This script downloads and configures the necessary model weights for:
1. Segment Anything Model (SAM) - ViT-B (375 MB)
2. Grounding DINO Swin-T Checkpoint (694 MB)
3. BERT Base Uncased Model & Tokenizer for Grounding DINO (400 MB)
4. Grounding DINO Config (Auto-generated)

Usage:
    python scripts/setup_weights.py
"""

import os
import sys
import urllib.request
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"

# URLs for weights
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
GD_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# Grounding DINO Swin-T Config Content
GD_CONFIG_CONTENT = """batch_size = 1
modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "/app/weights/bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
"""

def show_progress(block_num, block_size, total_size):
    """Callback function for urllib.request.urlretrieve to show download progress"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, int(downloaded * 100 / total_size))
        # Convert to MB
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r📥 Downloading... {percent}% | {downloaded_mb:.1f}/{total_mb:.1f} MB")
    else:
        sys.stdout.write(f"\r📥 Downloading... {downloaded / (1024 * 1024):.1f} MB")
    sys.stdout.flush()

def download_file(url, target_path):
    """Download a file with progress indicator"""
    print(f"\n🚀 Target: {target_path.name}")
    print(f"🔗 URL: {url}")
    
    if target_path.exists():
        # Check if file size is reasonable (not empty/corrupt)
        if target_path.stat().st_size > 1024 * 1024:
            print(f"✅ Already exists (Size: {target_path.stat().st_size / (1024*1024):.1f} MB). Skipping.")
            return True
        else:
            print("⚠️ Existing file is suspiciously small. Re-downloading.")
            target_path.unlink()
            
    try:
        urllib.request.urlretrieve(url, str(target_path), show_progress)
        print(f"\n🎉 Download completed: {target_path.name}")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {url}: {e}")
        return False

def write_gd_config():
    """Write the Grounding DINO config file"""
    config_path = WEIGHTS_DIR / "GroundingDINO_SwinT_OGC.py"
    print(f"\n🛠️ Creating config file: {config_path.relative_to(ROOT_DIR)}")
    try:
        with open(config_path, "w") as f:
            f.write(GD_CONFIG_CONTENT)
        print("✅ Config written successfully.")
        return True
    except Exception as e:
        print(f"❌ Error writing config: {e}")
        return False

def setup_bert_model():
    """Download and cache BERT model files using HuggingFace library"""
    bert_dir = WEIGHTS_DIR / "bert-base-uncased"
    print(f"\n🎓 Setting up BERT Base model in: {bert_dir.relative_to(ROOT_DIR)}")
    
    if bert_dir.exists() and any(bert_dir.iterdir()):
        print("✅ BERT model directory is not empty. Skipping auto-download.")
        return True
        
    print("⏳ Loading transformers to download and save 'bert-base-uncased' locally...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Download tokenizer and model
        print("📥 Pulling tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("📥 Pulling model weights...")
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Save local copy
        bert_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(bert_dir))
        model.save_pretrained(str(bert_dir))
        print("🎉 BERT model successfully cached locally!")
        return True
    except ImportError:
        print("⚠️ 'transformers' library not found. We will install it first if inside virtual env.")
        print("👉 Run: pip install transformers huggingface_hub")
        print("👉 Then rerun this script to cache BERT model.")
        return False
    except Exception as e:
        print(f"❌ Error downloading BERT model: {e}")
        return False

def main():
    print("=" * 60)
    print("🤖 Labelground Model Weights Downloader & Setup 🤖")
    print("=" * 60)
    
    # Ensure weights directory exists
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # Write config
    write_gd_config()
    
    # Download weights
    sam_success = download_file(SAM_URL, WEIGHTS_DIR / "sam_vit_b_01ec64.pth")
    gd_success = download_file(GD_URL, WEIGHTS_DIR / "groundingdino_swint_ogc.pth")
    
    # Setup BERT
    bert_success = setup_bert_model()
    
    print("\n" + "=" * 60)
    if sam_success and gd_success and bert_success:
        print("🎉 SUCCESS: All model weights and config are successfully set up!")
        print("🚀 You can now run the application offline with fully functional AI.")
    else:
        print("⚠️ WARNING: Setup is incomplete.")
        print("- SAM ViT-B:", "✅ Ready" if sam_success else "❌ Failed")
        print("- Grounding DINO:", "✅ Ready" if gd_success else "❌ Failed")
        print("- BERT Model:", "✅ Ready" if bert_success else "❌ Failed (Run: pip install transformers)")
    print("=" * 60)

if __name__ == "__main__":
    main()
