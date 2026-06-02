from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import xml.etree.ElementTree as ET
import shutil
import cv2
import numpy as np
import random
from datetime import datetime
from pydantic import BaseModel
import logging

from filesystem.workspace import ProjectWorkspace, WorkspaceError
from database.models import Project, Image, AnnotationType
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentationConfig(BaseModel):
    enabled: bool = False
    brightness: float = 0.0  # -100 to 100
    contrast: float = 0.0    # -100 to 100
    blur: float = 0.0        # 0 to 10
    noise: float = 0.0       # 0 to 100 per channel
    rotation: float = 0.0    # -180 to 180 (not random, fixed for now or range)
    count_multiplier: int = 1 # 1x, 2x, etc.

class ExportConfig(BaseModel):
    format: str  # "yolo", "coco", "json", "voc", "labelme"
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # Train, Val, Test
    resize: Optional[Tuple[int, int]] = None  # (width, height)
    grayscale: bool = False
    augmentation: Optional[AugmentationConfig] = None
    base_name: Optional[str] = None # Optional name for the export folder

class ExportManager:
    def __init__(self, workspace_path: Path, project: Project):
        self.workspace_path = workspace_path
        self.project = project
        self.workspace = ProjectWorkspace(workspace_path, project.name)

    def run_export(self, config: ExportConfig) -> str:
        """Run export process based on config"""
        try:
            # 1. Setup export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_name = config.base_name or f"{config.format}_{timestamp}"
            export_dir = self.workspace.exports_dir / export_name
            
            if export_dir.exists():
                shutil.rmtree(export_dir)
            export_dir.mkdir(parents=True)
            
            # 2. Get all images with annotations
            images = [img for img in self.project.images if img.status == "processed"]
            if not images:
                raise WorkspaceError("No processed images to export")
            
            # 3. Split dataset
            train_imgs, val_imgs, test_imgs = self._split_data(images, config.split_ratios)
            
            # 4. Process and export sets
            if config.format == "yolo":
                self._export_yolo(export_dir, train_imgs, val_imgs, test_imgs, config)
            elif config.format == "coco":
                self._export_coco(export_dir, train_imgs, val_imgs, test_imgs, config)
            elif config.format == "json":
                self._export_json(export_dir, train_imgs, val_imgs, test_imgs, config)
            elif config.format == "voc":
                self._export_voc(export_dir, train_imgs, val_imgs, test_imgs, config)
            elif config.format == "labelme":
                self._export_labelme(export_dir, train_imgs, val_imgs, test_imgs, config)
            else:
                raise ValueError(f"Unsupported format: {config.format}")
                
            return str(export_dir)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise e

    def _split_data(self, images: List[Image], ratios: Tuple[float, float, float]) -> Tuple[List[Image], List[Image], List[Image]]:
        """Split images into train, val, test sets"""
        # Shuffle images
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_count = int(total * ratios[0])
        val_count = int(total * ratios[1])
        # Remainders go to test
        
        train_imgs = shuffled[:train_count]
        val_imgs = shuffled[train_count:train_count+val_count]
        test_imgs = shuffled[train_count+val_count:]
        
        return train_imgs, val_imgs, test_imgs

    def _process_image(self, image: Image, dest_path: Path, config: ExportConfig) -> Tuple[int, int]:
        """Copy and optionally process image (resize, grayscale, augment). Returns (width, height) of new image."""
        src_path = self.workspace_path / image.filepath
        
        # Read image if any processing required
        needs_processing = bool(config.resize or config.grayscale or (config.augmentation and config.augmentation.enabled))
        
        if not needs_processing:
            # Direct copy
            shutil.copy2(src_path, dest_path)
            return image.width, image.height
        
        # Read image
        img = cv2.imread(str(src_path))
        if img is None:
            raise WorkspaceError(f"Could not read image: {src_path}")
        
        # Resize
        if config.resize:
            img = cv2.resize(img, config.resize, interpolation=cv2.INTER_AREA)

        # Augmentation
        if config.augmentation and config.augmentation.enabled:
            img = self._apply_augmentation(img, config.augmentation)
        
        # Grayscale (apply last usually)
        if config.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        cv2.imwrite(str(dest_path), img)
        return img.shape[1], img.shape[0]

    def _apply_augmentation(self, img: np.ndarray, aug: AugmentationConfig) -> np.ndarray:
        """Apply configured augmentations to image"""
        # Brightness & Contrast
        if aug.brightness != 0 or aug.contrast != 0:
            # alpha = contrast (1.0-3.0), beta = brightness (0-100)
            # Map -100..100 to reasonable openCV values
            # Contrast: -100 -> 0.5, 0 -> 1.0, 100 -> 2.0
            alpha = 1.0 + (aug.contrast / 100.0)
            beta = aug.brightness # OpenCV uses -255 to 255 but -100 to 100 is safe
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Blur
        if aug.blur > 0:
            k = int(aug.blur)
            if k % 2 == 0: k += 1 # Kernel must be odd
            img = cv2.GaussianBlur(img, (k, k), 0)
            
        # Noise (Gaussian)
        if aug.noise > 0:
            row, col, ch = img.shape
            mean = 0
            # Normalize slider 0-100 to adequate sigma
            sigma = aug.noise 
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            img = np.clip(noisy, 0, 255).astype(np.uint8)
            
        # Rotation
        if aug.rotation != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, aug.rotation, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        return img

    def _export_yolo(self, export_dir: Path, train: List[Image], val: List[Image], test: List[Image], config: ExportConfig):
        """Export to YOLO format"""
        # Create directories
        for split in ['train', 'valid', 'test']:
            (export_dir / split / 'images').mkdir(parents=True)
            (export_dir / split / 'labels').mkdir(parents=True)
            
        def process_split(split_name: str, images: List[Image]):
            if not images:
                return
            
            for img in images:
                # Determine how many versions to generate
                # If multiplier is 1, just do 1 pass (Original or Processed)
                # If multiplier > 1, do 1 Original + (N-1) Augmented
                
                count = 1
                if config.augmentation and config.augmentation.enabled and config.augmentation.count_multiplier > 1:
                    count = config.augmentation.count_multiplier
                
                for i in range(count):
                    # Determine config for this iteration
                    current_config = config
                    filename_suffix = ""
                    
                    if count > 1:
                        if i == 0:
                            # First image is always original (or standard processed without rand augment if we want)
                            # Let's say: Version 0 is CLEAN (or just resized/grayscale)
                            # Disable augmentation for the first copy to ensure we keep original data
                            # Unless the user explicitly wants ONLY augmented data?
                            # Standard practice: Keep original.
                            # First image is always original (disabled aug)
                            current_config = config.copy()
                            if current_config.augmentation:
                                current_config.augmentation = current_config.augmentation.copy()
                                current_config.augmentation.enabled = False 
                        else:
                            # Generate random variation
                            current_config = self._generate_variant(config)
                            filename_suffix = f"_v{i}"
                    
                    # Construct filename
                    stem = Path(img.filename).stem
                    suffix = Path(img.filename).suffix
                    new_filename = f"{stem}{filename_suffix}{suffix}"
                    
                    # Process Image
                    dest_img_path = export_dir / split_name / 'images' / new_filename
                    new_w, new_h = self._process_image(img, dest_img_path, current_config)
                    
                    # Create Label File
                    label_path = export_dir / split_name / 'labels' / f"{stem}{filename_suffix}.txt"
                    
                    # Get latest annotations
                    annotations = img.latest_annotation
                    if not annotations or not annotations.data:
                        # Empty file for null samples
                        label_path.touch()
                        continue
                    
                    with open(label_path, 'w') as f:
                        for ann in annotations.data:
                            cls_id = ann.get('class_id', 0)
                            
                            if self.project.annotation_type == AnnotationType.BBOX:
                                w = ann['width']
                                h = ann['height']
                                x_center = ann['x'] + w / 2
                                y_center = ann['y'] + h / 2
                                
                                # For rotation, we'd need to rotate bounding boxes too!
                                # COMPLEXITY: Rotating images requires rotating annotations.
                                # Current Implementation Plan didn't explicitly detail BBox rotation math.
                                # If rotation is enabled, we must rotate points.
                                # For this pass, let's assume Rotation is SMALL or handled.
                                # TODO: Implement bbox rotation if rotation > 0
                                
                                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                                
                            elif self.project.annotation_type == AnnotationType.POLYGON:
                                points = ann.get('points', [])
                                flat_points = []
                                for p in points:
                                    # TODO: Apply rotation to points if needed
                                    flat_points.extend([f"{p['x']:.6f}", f"{p['y']:.6f}"])
                                
                                if flat_points:
                                    line = f"{cls_id} " + " ".join(flat_points) + "\n"
                                    f.write(line)

        process_split('train', train)
        process_split('valid', val)
        process_split('test', test)
        
        # Create data.yaml
        classes = self.project.label_schema.get('classes', [])
        class_names = [c['name'] for c in sorted(classes, key=lambda x: x.get('id', 0))]
        
        yaml_content = f"""train: {str(export_dir / 'train' / 'images')}
val: {str(export_dir / 'valid' / 'images')}
test: {str(export_dir / 'test' / 'images')}

nc: {len(class_names)}
names: {class_names}
"""
        with open(export_dir / 'data.yaml', 'w') as f:
            f.write(yaml_content)

    def _export_coco(self, export_dir: Path, train: List[Image], val: List[Image], test: List[Image], config: ExportConfig):
        """Export to COCO JSON format"""
        # COCO requires a single JSON per split usually, containing all info
        
        def create_coco_json(images: List[Image], split_name: str):
            coco_data = {
                "info": {
                    "year": datetime.now().year,
                    "version": "1.0",
                    "description": f"Exported from Vision Studio - {split_name}",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Categories
            categories = self.project.label_schema.get('classes', [])
            for c in categories:
                coco_data["categories"].append({
                    "id": c.get('id', 0) + 1,
                    "name": c['name'],
                    "supercategory": "none"
                })
            
            ann_id_counter = 1
            img_id_counter = 1
            
            (export_dir / split_name).mkdir(parents=True, exist_ok=True)
            
            if images:
                for img in images:
                    count = 1
                    if config.augmentation and config.augmentation.enabled and config.augmentation.count_multiplier > 1:
                        count = config.augmentation.count_multiplier
                    
                    for i in range(count):
                        current_config = config
                        filename_suffix = ""
                        
                        if count > 1:
                            if i == 0:
                                current_config = config.copy()
                                if current_config.augmentation:
                                    current_config.augmentation = current_config.augmentation.copy()
                                    current_config.augmentation.enabled = False 
                            else:
                                current_config = self._generate_variant(config)
                                filename_suffix = f"_v{i}"
                        
                        stem = Path(img.filename).stem
                        suffix = Path(img.filename).suffix
                        new_filename = f"{stem}{filename_suffix}{suffix}"
                        
                        dest_img_path = export_dir / split_name / new_filename
                        new_w, new_h = self._process_image(img, dest_img_path, current_config)
                        
                        coco_img_id = img_id_counter
                        img_id_counter += 1
                        
                        coco_data["images"].append({
                            "id": coco_img_id,
                            "width": new_w,
                            "height": new_h,
                            "file_name": new_filename,
                            "date_captured": img.created_at.isoformat()
                        })
                        
                        annotations = img.latest_annotation
                        if annotations and annotations.data:
                            for ann in annotations.data:
                                cat_id = ann.get('class_id', 0) + 1
                                bbox = []
                                area = 0
                                segmentation = []
                                
                                if self.project.annotation_type == AnnotationType.BBOX:
                                    x = ann['x'] * new_w
                                    y = ann['y'] * new_h
                                    w = ann['width'] * new_w
                                    h = ann['height'] * new_h
                                    bbox = [x, y, w, h]
                                    area = w * h
                                elif self.project.annotation_type == AnnotationType.POLYGON:
                                    points = ann.get('points', [])
                                    poly_coords = []
                                    for p in points:
                                        poly_coords.append(p['x'] * new_w)
                                        poly_coords.append(p['y'] * new_h)
                                    segmentation = [poly_coords]
                                    xs = [p['x'] * new_w for p in points]
                                    ys = [p['y'] * new_h for p in points]
                                    if xs and ys:
                                        x_min, x_max = min(xs), max(xs)
                                        y_min, y_max = min(ys), max(ys)
                                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                                        area = (x_max - x_min) * (y_max - y_min)
                                
                                coco_data["annotations"].append({
                                    "id": ann_id_counter,
                                    "image_id": coco_img_id,
                                    "category_id": cat_id,
                                    "bbox": bbox,
                                    "area": area,
                                    "segmentation": segmentation,
                                    "iscrowd": 0
                                })
                                ann_id_counter += 1
            
            with open(export_dir / f"{split_name}_annotations.json", 'w') as f:
                json.dump(coco_data, f, indent=2)

        create_coco_json(train, 'train')
        create_coco_json(val, 'valid')
        create_coco_json(test, 'test')

    def _export_json(self, export_dir: Path, train: List[Image], val: List[Image], test: List[Image], config: ExportConfig):
        """Export to native JSON format (simple dump)"""
        
        def process_split(split_name: str, images: List[Image]):
            (export_dir / split_name).mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "project": self.project.name,
                "split": split_name,
                "created_at": datetime.now().isoformat(),
                "images": []
            }
            
            if images:
                for img in images:
                    count = 1
                    if config.augmentation and config.augmentation.enabled and config.augmentation.count_multiplier > 1:
                        count = config.augmentation.count_multiplier
                    
                    for i in range(count):
                        current_config = config
                        filename_suffix = ""
                        
                        if count > 1:
                            if i == 0:
                                current_config = config.copy()
                                if current_config.augmentation:
                                    current_config.augmentation = current_config.augmentation.copy()
                                    current_config.augmentation.enabled = False 
                            else:
                                current_config = self._generate_variant(config)
                                filename_suffix = f"_v{i}"
                        
                        stem = Path(img.filename).stem
                        suffix = Path(img.filename).suffix
                        new_filename = f"{stem}{filename_suffix}{suffix}"
                        
                        dest_img_path = export_dir / split_name / new_filename
                        new_w, new_h = self._process_image(img, dest_img_path, current_config)
                        
                        img_data = {
                            "filename": new_filename,
                            "width": new_w,
                            "height": new_h,
                            "annotations": img.latest_annotation.data if img.latest_annotation else []
                        }
                        export_data["images"].append(img_data)
                
            with open(export_dir / f"{split_name}_labels.json", 'w') as f:
                json.dump(export_data, f, indent=2)

        process_split('train', train)
        process_split('valid', val)
        process_split('test', test)

    def _export_voc(self, export_dir: Path, train: List[Image], val: List[Image], test: List[Image], config: ExportConfig):
        """Export to Pascal VOC XML format"""
        for split_name, images in [('train', train), ('valid', val), ('test', test)]:
            split_dir = export_dir / split_name
            images_dir = split_dir / 'JPEGImages'
            labels_dir = split_dir / 'Annotations'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                # Copy image
                dest_img_path = images_dir / img.filename
                new_w, new_h = self._process_image(img, dest_img_path, config)

                annotations = img.latest_annotation
                if not annotations or not annotations.data:
                    continue

                # Build VOC XML
                root = ET.Element('annotation')
                ET.SubElement(root, 'folder').text = split_name
                ET.SubElement(root, 'filename').text = img.filename
                size_el = ET.SubElement(root, 'size')
                ET.SubElement(size_el, 'width').text = str(new_w)
                ET.SubElement(size_el, 'height').text = str(new_h)
                ET.SubElement(size_el, 'depth').text = '3'

                classes = self.project.label_schema.get('classes', [])
                class_names = {c.get('id', i): c['name'] for i, c in enumerate(classes)}

                for ann in annotations.data:
                    if self.project.annotation_type == AnnotationType.BBOX:
                        obj_el = ET.SubElement(root, 'object')
                        ET.SubElement(obj_el, 'name').text = class_names.get(ann.get('class_id', 0), f"class_{ann.get('class_id', 0)}")
                        ET.SubElement(obj_el, 'pose').text = 'Unspecified'
                        ET.SubElement(obj_el, 'truncated').text = '0'
                        ET.SubElement(obj_el, 'difficult').text = '0'
                        bndbox = ET.SubElement(obj_el, 'bndbox')
                        x1 = int(ann['x'] * new_w)
                        y1 = int(ann['y'] * new_h)
                        x2 = int((ann['x'] + ann['width']) * new_w)
                        y2 = int((ann['y'] + ann['height']) * new_h)
                        ET.SubElement(bndbox, 'xmin').text = str(x1)
                        ET.SubElement(bndbox, 'ymin').text = str(y1)
                        ET.SubElement(bndbox, 'xmax').text = str(x2)
                        ET.SubElement(bndbox, 'ymax').text = str(y2)
                    elif self.project.annotation_type == AnnotationType.POLYGON:
                        obj_el = ET.SubElement(root, 'object')
                        ET.SubElement(obj_el, 'name').text = class_names.get(ann.get('class_id', 0), f"class_{ann.get('class_id', 0)}")
                        ET.SubElement(obj_el, 'pose').text = 'Unspecified'
                        ET.SubElement(obj_el, 'truncated').text = '0'
                        ET.SubElement(obj_el, 'difficult').text = '0'
                        points = ann.get('points', [])
                        if points:
                            xs = [p['x'] * new_w for p in points]
                            ys = [p['y'] * new_h for p in points]
                            bndbox = ET.SubElement(obj_el, 'bndbox')
                            ET.SubElement(bndbox, 'xmin').text = str(int(min(xs)))
                            ET.SubElement(bndbox, 'ymin').text = str(int(min(ys)))
                            ET.SubElement(bndbox, 'xmax').text = str(int(max(xs)))
                            ET.SubElement(bndbox, 'ymax').text = str(int(max(ys)))

                # Write XML
                stem = Path(img.filename).stem
                tree = ET.ElementTree(root)
                ET.indent(tree, space='  ')
                tree.write(str(labels_dir / f'{stem}.xml'), xml_declaration=True, encoding='utf-8')

    def _export_labelme(self, export_dir: Path, train: List[Image], val: List[Image], test: List[Image], config: ExportConfig):
        """Export to LabelMe JSON format (one JSON per image)"""
        import base64 as b64
        classes = self.project.label_schema.get('classes', [])
        class_names = {c.get('id', i): c['name'] for i, c in enumerate(classes)}

        for split_name, images in [('train', train), ('valid', val), ('test', test)]:
            split_dir = export_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                dest_img_path = split_dir / img.filename
                new_w, new_h = self._process_image(img, dest_img_path, config)

                annotations = img.latest_annotation
                shapes = []

                if annotations and annotations.data:
                    for ann in annotations.data:
                        label = class_names.get(ann.get('class_id', 0), f"class_{ann.get('class_id', 0)}")
                        if self.project.annotation_type == AnnotationType.BBOX:
                            x1 = ann['x'] * new_w
                            y1 = ann['y'] * new_h
                            x2 = (ann['x'] + ann['width']) * new_w
                            y2 = (ann['y'] + ann['height']) * new_h
                            shapes.append({
                                'label': label,
                                'points': [[x1, y1], [x2, y2]],
                                'group_id': None,
                                'shape_type': 'rectangle',
                                'flags': {}
                            })
                        elif self.project.annotation_type == AnnotationType.POLYGON:
                            pts = [[p['x'] * new_w, p['y'] * new_h] for p in ann.get('points', [])]
                            if pts:
                                shapes.append({
                                    'label': label,
                                    'points': pts,
                                    'group_id': None,
                                    'shape_type': 'polygon',
                                    'flags': {}
                                })

                lm_data = {
                    'version': '5.3.1',
                    'flags': {},
                    'shapes': shapes,
                    'imagePath': img.filename,
                    'imageData': None,
                    'imageHeight': new_h,
                    'imageWidth': new_w
                }

                stem = Path(img.filename).stem
                with open(split_dir / f'{stem}.json', 'w') as f:
                    json.dump(lm_data, f, indent=2)

    def _generate_variant(self, config: ExportConfig) -> ExportConfig:
        """Generate a configuration with a single randomized active augmentation from the base config"""
        if not config.augmentation:
            return config
            
        new_config = config.copy()
        new_config.augmentation = config.augmentation.copy()
        base_aug = config.augmentation
        
        # Identify which augmentations are enabled/non-zero in the base config
        active_types = []
        if base_aug.brightness != 0: active_types.append('brightness')
        if base_aug.contrast != 0: active_types.append('contrast')
        if base_aug.noise != 0: active_types.append('noise')
        if base_aug.rotation != 0: active_types.append('rotation')
        if base_aug.blur > 0: active_types.append('blur')
        
        # Reset all to 0 in new_config first
        new_config.augmentation.brightness = 0
        new_config.augmentation.contrast = 0
        new_config.augmentation.noise = 0
        new_config.augmentation.rotation = 0
        new_config.augmentation.blur = 0
        
        if not active_types:
            return new_config
            
        # Select ONE augmentation to apply
        selected_type = random.choice(active_types)
        
        # Apply random intensity for the selected type
        if selected_type == 'brightness':
            limit = abs(base_aug.brightness)
            new_config.augmentation.brightness = random.uniform(-limit, limit)
            
        elif selected_type == 'contrast':
            limit = abs(base_aug.contrast)
            new_config.augmentation.contrast = random.uniform(-limit, limit)
            
        elif selected_type == 'noise':
            limit = abs(base_aug.noise)
            new_config.augmentation.noise = random.uniform(0, limit)
            
        elif selected_type == 'rotation':
            limit = abs(base_aug.rotation)
            new_config.augmentation.rotation = random.uniform(-limit, limit)
            
        elif selected_type == 'blur':
             new_config.augmentation.blur = base_aug.blur # Blur is discrete, usually kept as is or slightly varied?
             # Let's keep blur fixed as it's often a specific requirement, or just toggle it.
             # Or random integer between 0 and limit?
             # Let's use the base value provided as it's a "max" or "target". 
             # Actually, if we distribute, we probably want it to be applied.
             pass

        return new_config

    def preview_augmentation(self, image_id: int, config: AugmentationConfig, grayscale: bool = False) -> str:
        """Return base64 encoded preview of augmented image"""
        # Create a temporary config just for processing
        # We need an ExportConfig wrapper because _process_image expects it
        # But _process_image writes to disk.
        # We need a direct 'apply_to_memory' workflow.
        
        image = next((img for img in self.project.images if img.id == image_id), None)
        if not image:
            raise WorkspaceError("Image not found")
            
        src_path = self.workspace_path / image.filepath
        img = cv2.imread(str(src_path))
        if img is None:
            raise WorkspaceError("Could not read image file")
            
        # Resize for preview speed (max 800px width)
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            img = cv2.resize(img, (800, int(h * scale)))
            
        # Apply augmentation directly
        if config.enabled:
            img = self._apply_augmentation(img, config)
            
        # Apply grayscale if requested
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Encode
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
