import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import supervision as sv

# Attempt imports for AI libraries
try:
    from groundingdino.util.inference import load_model, load_image, predict
    import groundingdino.datasets.transforms as T
    from segment_anything import SamPredictor, sam_model_registry
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

class AIService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIService, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.models = {}
        return cls._instance

    def initialize_models(self):
        if self.initialized:
            return
        
        if not AI_AVAILABLE:
            print("⚠️ AI Libraries (GroundingDINO/SAM) not found. Auto-annotation will be disabled.")
            return

        weights_dir = Path("weights")
        gd_config = weights_dir / "GroundingDINO_SwinT_OGC.py"
        gd_checkpoint = weights_dir / "groundingdino_swint_ogc.pth"
        sam_checkpoint = weights_dir / "sam_vit_b_01ec64.pth"

        try:
            print(f"🔍 Initializing models. Device: {self.device}")
            weights_dir = Path("weights").absolute()
            print(f"📂 Weights directory: {weights_dir}")
            
            gd_config = weights_dir / "GroundingDINO_SwinT_OGC.py"
            gd_checkpoint = weights_dir / "groundingdino_swint_ogc.pth"
            sam_checkpoint = weights_dir / "sam_vit_b_01ec64.pth"

            # Check if paths exist
            if not gd_config.exists(): print(f"❌ Missing GD config: {gd_config}")
            if not gd_checkpoint.exists(): print(f"❌ Missing GD checkpoint: {gd_checkpoint}")
            if not sam_checkpoint.exists(): print(f"❌ Missing SAM checkpoint: {sam_checkpoint}")

            # Load GroundingDINO
            if gd_config.exists() and gd_checkpoint.exists():
                self.models['grounding_dino'] = load_model(
                    str(gd_config), 
                    str(gd_checkpoint), 
                    device=self.device
                )
                print(f"✅ GroundingDINO loaded on {self.device}")
            
            # Load SAM
            if sam_checkpoint.exists():
                sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
                sam.to(device=self.device)
                self.models['sam_predictor'] = SamPredictor(sam)
                print(f"✅ SAM loaded on {self.device}")
                
            self.initialized = True
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
            import traceback
            traceback.print_exc()

    def run_grounding_dino(self, image_path: str, caption: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        if 'grounding_dino' not in self.models:
            return np.array([]), np.array([]), []

        # Load image for GroundingDINO
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=self.models['grounding_dino'],
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), []
            
        # Convert relative [xc, yc, w, h] to absolute [x1, y1, x2, y2]
        h, w, _ = image_source.shape
        boxes_xyxy = sv.xcycwh_to_xyxy(boxes.numpy()) * [w, h, w, h]
        
        return boxes_xyxy, logits.numpy(), phrases

    def run_sam_refinement(self, image_cv, boxes_xyxy):
        if 'sam_predictor' not in self.models or len(boxes_xyxy) == 0:
            return None

        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.models['sam_predictor'].set_image(image_rgb)
        
        masks = []
        for box in boxes_xyxy:
            input_box = np.array(box)
            m, scores, _ = self.models['sam_predictor'].predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks.append(m[0])
            
        return np.stack(masks) if masks else None

    async def auto_annotate(self, image_path: str, classes: List[Dict]) -> List[Dict]:
        """
        Runs auto-annotation on a single image.
        classes: List of { id: int, name: str, prompt: str, threshold: float }
        """
        if not self.initialized:
            self.initialize_models()
        
        if not self.initialized:
            print("⚠️ AIService not initialized, skipping auto-annotate")
            return []

        print(f"🖼️ Auto-annotating image: {image_path}")
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            print(f"❌ Could not read image: {image_path}")
            return []
            
        h_img, w_img = image_cv.shape[:2]
        all_annotations = []

        for cls in classes:
            # GroundingDINO works best if the prompt is lowercase and ends with a dot
            raw_prompt = cls.get('prompt', cls['name'])
            prompt = raw_prompt.lower().strip()
            if not prompt.endswith('.'):
                prompt += " ."
            
            threshold = cls.get('threshold', 0.35)
            print(f"🎯 Running detection for class '{cls['name']}' with prompt '{prompt}' @ {threshold}")
            
            try:
                # Run DINO
                boxes, logits, phrases = self.run_grounding_dino(image_path, prompt, box_threshold=threshold)
                print(f"   Found {len(boxes)} raw detections for '{cls['name']}'")
                
                if len(boxes) == 0:
                    continue
                    
                # Run SAM if available
                masks = self.run_sam_refinement(image_cv, boxes)
                if masks is not None:
                    print(f"   SAM refined {len(masks)} masks")
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    ann = {
                        "class_id": cls['id'],
                        "confidence": float(logits[i]),
                        "type": "bbox",
                        "x": float(box[0] / w_img),
                        "y": float(box[1] / h_img),
                        "width": float((box[2] - box[0]) / w_img),
                        "height": float((box[3] - box[1]) / h_img)
                    }
                    
                    # If we have a mask, convert to polygon
                    if masks is not None:
                        mask = masks[i]
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Largest contour
                            cnt = max(contours, key=cv2.contourArea)
                            # Simplify contour
                            epsilon = 0.002 * cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, epsilon, True)
                            
                            points = []
                            for p in approx:
                                points.append({
                                    "x": float(p[0][0] / w_img),
                                    "y": float(p[0][1] / h_img)
                                })
                            
                            if len(points) >= 3:
                                ann["type"] = "polygon"
                                ann["points"] = points
                                
                    all_annotations.append(ann)
            except Exception as e:
                print(f"❌ Error during AI processing for class {cls['name']}: {e}")
                import traceback
                traceback.print_exc()

        return all_annotations

# Singleton instance
ai_service = AIService()
