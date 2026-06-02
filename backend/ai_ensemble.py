"""
AI Ensemble Service for Auto-Annotation

This module provides a unified interface for running multiple AI models
(YOLO-World and Grounding DINO) and merging their predictions using NMS.
"""

import os

import logging
logger = logging.getLogger(__name__)

# Point all AI cache directories to the 'weights' folder for permanent storage
weights_dir = "/app/weights"
os.makedirs(weights_dir, exist_ok=True)

os.environ['HOME'] = weights_dir
os.environ['YOLO_CONFIG_DIR'] = os.path.join(weights_dir, 'yolo_config')
os.environ['MPLCONFIGDIR'] = os.path.join(weights_dir, 'matplotlib')
os.environ['HF_HOME'] = os.path.join(weights_dir, 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(weights_dir, 'torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(weights_dir, 'cache')

logger.info(f"📂 AI Weights & Cache permanently stored in: {weights_dir}")

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
# Try importing AI libraries
YOLO_WORLD_AVAILABLE = False
GROUNDING_DINO_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_WORLD_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics not available. YOLO-World will be disabled.")

try:
    from groundingdino.util.inference import load_model, load_image, predict
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    logger.warning("GroundingDINO not available. Grounding DINO will be disabled.")

SAM_AVAILABLE = False
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    logger.warning("SAM not available. Segmentation refinement will be disabled.")

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger.warning("supervision not available. NMS will use fallback.")


class AIEnsembleService:
    """
    Unified AI service that runs both YOLO-World and Grounding DINO,
    then merges predictions using Non-Maximum Suppression (NMS).
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIEnsembleService, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.models = {}
        return cls._instance
    
    def get_weights_dir(self) -> Path:
        """Get the weights directory path"""
        return Path(__file__).parent.parent / "weights"
    
    def initialize_models(self, force: bool = False):
        """Initialize all available AI models"""
        if self.initialized and not force:
            return
        
        weights_dir = self.get_weights_dir()
        weights_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔍 Initializing AI models. Device: {self.device}")
        logger.info(f"📂 Weights directory: {weights_dir}")
        
        # Initialize YOLO-World (Large variant for best accuracy)
        if YOLO_WORLD_AVAILABLE:
            try:
                yolo_world_path = weights_dir / "yolov8l-worldv2.pt"
                
                if not yolo_world_path.exists():
                    logger.info("⬇️ Downloading YOLO-World Large model...")
                    # YOLO will auto-download if we just load it
                    model = YOLO("yolov8l-worldv2.pt")
                    # Move weights to our directory
                    import shutil
                    default_path = Path.home() / ".cache" / "ultralytics" / "yolov8l-worldv2.pt"
                    if default_path.exists():
                        shutil.copy2(default_path, yolo_world_path)
                else:
                    model = YOLO(str(yolo_world_path))
                
                self.models['yolo_world'] = model
                logger.info(f"✅ YOLO-World Large loaded on {self.device}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load YOLO-World: {e}")
        
        # Initialize Grounding DINO
        if GROUNDING_DINO_AVAILABLE:
            try:
                gd_config = weights_dir / "GroundingDINO_SwinT_OGC.py"
                gd_checkpoint = weights_dir / "groundingdino_swint_ogc.pth"
                
                if gd_config.exists() and gd_checkpoint.exists():
                    self.models['grounding_dino'] = load_model(
                        str(gd_config),
                        str(gd_checkpoint),
                        device=self.device
                    )
                    logger.info(f"✅ Grounding DINO loaded on {self.device}")
                else:
                    logger.warning("⚠️ Grounding DINO weights not found")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load Grounding DINO: {e}")
        
        # Initialize SAM for segmentation refinement
        if SAM_AVAILABLE:
            try:
                sam_checkpoint = weights_dir / "sam_vit_b_01ec64.pth"
                if sam_checkpoint.exists():
                    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
                    sam.to(device=self.device)
                    self.models['sam_predictor'] = SamPredictor(sam)
                    logger.info(f"✅ SAM loaded on {self.device}")
                else:
                    logger.warning(f"⚠️ SAM checkpoint not found at {sam_checkpoint}")
            except Exception as e:
                logger.error(f"❌ Failed to load SAM: {e}")
        
        self.initialized = True
        logger.info(f"📊 Models loaded: {list(self.models.keys())}")
    
    def run_yolo_world(
        self, 
        image_path: str, 
        classes: List[str],
        confidence: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLO-World inference.
        
        Returns:
            boxes_xyxy: Array of [x1, y1, x2, y2] boxes
            confidences: Array of confidence scores
            detected_classes: List of detected class names
        """
        if 'yolo_world' not in self.models:
            return np.array([]), np.array([]), []
        
        model = self.models['yolo_world']
        
        # Set custom classes for open-vocabulary detection
        model.set_classes(classes)
        
        # Run inference
        results = model.predict(image_path, conf=confidence, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.array([]), np.array([]), []
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        return boxes, confidences, class_ids
    
    def run_grounding_dino(
        self,
        image_path: str,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run Grounding DINO inference.
        
        Returns:
            boxes_xyxy: Array of [x1, y1, x2, y2] boxes
            confidences: Array of confidence scores
            detected_phrases: List of detected phrases
        """
        if 'grounding_dino' not in self.models:
            return np.array([]), np.array([]), []
        
        # Load image for GroundingDINO
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=self.models['grounding_dino'],
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), []
        
        # Convert relative [xc, yc, w, h] to absolute [x1, y1, x2, y2]
        h, w, _ = image_source.shape
        
        if SUPERVISION_AVAILABLE:
            boxes_xyxy = sv.xcycwh_to_xyxy(boxes.numpy()) * [w, h, w, h]
        else:
            # Manual conversion
            boxes_np = boxes.numpy()
            boxes_xyxy = np.zeros_like(boxes_np)
            boxes_xyxy[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * w
            boxes_xyxy[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * h
            boxes_xyxy[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * w
            boxes_xyxy[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * h
        
        return boxes_xyxy, logits.numpy(), phrases
    
    def apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        Keeps the highest confidence box when IoU > threshold.
        """
        if len(boxes) == 0:
            return boxes, scores, class_ids
        
        # Use OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        indices = indices.flatten()
        return boxes[indices], scores[indices], class_ids[indices]
    
        return boxes, confidences, class_ids

    def run_sam(
        self, 
        image: np.ndarray, 
        boxes_xyxy: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> List[Optional[List[Dict[str, float]]]]:
        """
        Run SAM on the given image and prompts (boxes or points) to generate polygons.
        Returns a list of polygons, where each polygon is a list of normalized {x, y} dicts.
        """
        if not self.initialized:
            self.initialize_models()

        if 'sam_predictor' not in self.models:
            logger.warning("⚠️ SAM Predictor not available. Returning None for polygons.")
            return []
            
        predictor = self.models['sam_predictor']
        predictor.set_image(image)
        
        h_img, w_img = image.shape[:2]
        polygons = []
        
        # Determine number of requests
        num_requests = 0
        if boxes_xyxy is not None:
            num_requests = len(boxes_xyxy)
        elif points is not None:
            # If it's a single click for one object
            num_requests = 1
        
        for i in range(num_requests):
            try:
                curr_box = None
                curr_points = None
                curr_labels = None
                
                if boxes_xyxy is not None:
                    box = boxes_xyxy[i]
                    curr_box = np.array([
                        max(0, box[0]), max(0, box[1]),
                        min(w_img, box[2]), min(h_img, box[3])
                    ])
                
                if points is not None:
                    curr_points = points
                    curr_labels = labels if labels is not None else np.ones(len(points))

                masks, scores, logits = predictor.predict(
                    point_coords=curr_points,
                    point_labels=curr_labels,
                    box=curr_box,
                    multimask_output=False
                )
                mask = masks[0]
                
                # Convert mask to polygon using OpenCV
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Approximate polygon to reduce point count
                    # Using slightly larger epsilon (0.005 instead of 0.001) for better usability in manual editing
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # Convert to normalized list of points
                    points_out = []
                    for pt in approx:
                        points_out.append({
                            "x": float(max(0.0, min(1.0, pt[0][0] / w_img))),
                            "y": float(max(0.0, min(1.0, pt[0][1] / h_img)))
                        })
                    polygons.append(points_out)
                else:
                    polygons.append(None)
            except Exception as e:
                logger.error(f"❌ SAM segmentation failed for request {i}: {e}")
                polygons.append(None)
                
        return polygons

    async def auto_annotate_ensemble(
        self,
        image_path: str,
        classes: List[Dict],
        use_yolo_world: bool = True,
        use_grounding_dino: bool = True,
        nms_iou_threshold: float = 0.5,
        min_confidence: float = 0.3,
        custom_model_path: str = None,
        project_name: str = "Unknown",
        output_type: str = "bbox"
    ) -> Dict:
        """
        Run ensemble auto-annotation using both models.
        
        Args:
            image_path: Path to the image
            classes: List of {id, name, prompt, threshold} dicts
            use_yolo_world: Whether to use YOLO-World
            use_grounding_dino: Whether to use Grounding DINO
            nms_iou_threshold: IoU threshold for NMS
            min_confidence: Minimum confidence to keep
            
        Returns:
            List of annotation dicts
        """
        if not self.initialized:
            self.initialize_models()
        
        if not self.initialized:
            logger.warning("⚠️ AI models not initialized")
            return []
        
        # Read image for dimensions
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"[Project: {project_name}] ❌ Could not read image: {image_path}")
            return []
        
        h_img, w_img = img.shape[:2]
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_sources = []  # Track which model produced the box
        
        # Prepare prompts for models
        prompts = [c.get('prompt') or c['name'] for c in classes]
        # Map each prompt index back to the real class ID
        index_to_class_id = {i: c['id'] for i, c in enumerate(classes)}
        
        # 1. Run Custom Project Model if available
        if custom_model_path and Path(custom_model_path).exists():
            logger.info(f"[Project: {project_name}] 🎓 Running CUSTOM model on {Path(image_path).name}")
            try:
                boxes, scores, class_ids = self.run_custom_yolo(
                    image_path,
                    custom_model_path,
                    confidence=min_confidence
                )
                for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                    all_boxes.append(box)
                    all_scores.append(float(score))
                    all_class_ids.append(int(cls_id))
                    all_sources.append('custom_model')
            except Exception as e:
                logger.error(f"❌ Custom model inference failed: {e}")

        # Only run zero-shot models if no custom model boxes were found, 
        # or if we want to combine them (here we combine for maximum coverage)
        
        # Run YOLO-World
        if use_yolo_world and 'yolo_world' in self.models:
            logger.info(f"[Project: {project_name}] 🚀 Running YOLO-World on {Path(image_path).name}")
            try:
                boxes, scores, detected_indices = self.run_yolo_world(
                    image_path, 
                    prompts,
                    confidence=min_confidence
                )
                
                for i, (box, score, idx) in enumerate(zip(boxes, scores, detected_indices)):
                    cls_id = index_to_class_id.get(idx, 0)
                    all_boxes.append(box)
                    all_scores.append(float(score))
                    all_class_ids.append(cls_id)
                    all_sources.append('yolo_world')
                    
                logger.info(f"   YOLO-World found {len(boxes)} detections mapping to project classes")
                
            except Exception as e:
                logger.error(f"❌ YOLO-World error: {e}")
        
        # Run Grounding DINO for each class
        if use_grounding_dino and 'grounding_dino' in self.models:
            logger.info(f"[Project: {project_name}] 🔍 Running Grounding DINO on {Path(image_path).name}")
            try:
                for cls in classes:
                    prompt = cls.get('prompt', cls['name']).lower().strip()
                    if not prompt.endswith('.'):
                        prompt += " ."
                    
                    threshold = cls.get('threshold', 0.35)
                    
                    boxes, scores, phrases = self.run_grounding_dino(
                        image_path,
                        prompt,
                        box_threshold=threshold
                    )
                    
                    for box, score in zip(boxes, scores):
                        all_boxes.append(box)
                        all_scores.append(float(score))
                        all_class_ids.append(cls['id'])
                        all_sources.append('grounding_dino')
                
                logger.info(f"   Grounding DINO found {sum(1 for s in all_sources if s == 'grounding_dino')} detections")
                
            except Exception as e:
                logger.error(f"❌ Grounding DINO error: {e}")
        
        if len(all_boxes) == 0:
            return []
        
        # Convert to numpy arrays
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_class_ids = np.array(all_class_ids)
        
        # Apply NMS to merge overlapping boxes
        nms_boxes, nms_scores, nms_class_ids = self.apply_nms(
            all_boxes, all_scores, all_class_ids, nms_iou_threshold
        )
        
        logger.info(f"[Project: {project_name}] 📊 After NMS: {len(nms_boxes)} boxes (from {len(all_boxes)})")
        
        # Ensure values are strictly between 0 and 1
        def clamp(v): return max(0.0, min(1.0, float(v)))

        # Run segmentation if polygon output requested
        polygons = None
        if output_type == "polygon" and 'sam_predictor' in self.models:
            logger.info(f"[Project: {project_name}] 🧩 Running SAM segmentation for {len(nms_boxes)} objects")
            polygons = self.run_sam(img, nms_boxes)

        annotations = []
        for i in range(len(nms_boxes)):
            box = nms_boxes[i]
            
            # Create base annotation
            ann = {
                "class_id": int(nms_class_ids[i]),
                "confidence": float(nms_scores[i]),
                "source": "ai_ensemble"
            }
            
            # Add geometry based on output type
            if output_type == "polygon" and polygons and polygons[i]:
                ann["type"] = "polygon"
                ann["points"] = polygons[i]
                # Still include bbox as metadata/fallback
                ann["x"] = clamp(box[0] / w_img)
                ann["y"] = clamp(box[1] / h_img)
                ann["width"] = clamp((box[2] - box[0]) / w_img)
                ann["height"] = clamp((box[3] - box[1]) / h_img)
            else:
                ann["type"] = "bbox"
                ann["x"] = clamp(box[0] / w_img)
                ann["y"] = clamp(box[1] / h_img)
                ann["width"] = clamp((box[2] - box[0]) / w_img)
                ann["height"] = clamp((box[3] - box[1]) / h_img)
                
            annotations.append(ann)
        
        return {
            "annotations": annotations,
            "models_used": list(set(all_sources)),
            "custom_model_active": custom_model_path is not None and Path(custom_model_path).exists()
        }
    
    def get_available_models(self) -> List[str]:
        """Return list of available model names"""
        return list(self.models.keys())
    
    def is_ready(self) -> bool:
        """Check if at least one model is ready"""
        return len(self.models) > 0

    def clear_model_cache(self, model_path: str):
        """Force a reload of a specific model from disk"""
        cache_key = f"custom_{model_path}"
        if cache_key in self.models:
            del self.models[cache_key]
            logger.info(f"♻️ Model cache cleared for: {model_path}")


# Singleton instance
ai_ensemble = AIEnsembleService()
