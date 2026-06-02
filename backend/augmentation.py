"""
Data Augmentation Module

Provides functions to generate augmented versions of images and their 
associated annotations (bounding boxes, polygons) for training expansion.
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

class DataAugmentor:
    """
    Handles image and annotation transformations to expand datasets.
    """
    
    @staticmethod
    def apply_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale but keep 3 channels"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_flip(image: np.ndarray, annotations: List[Dict], horizontal=True) -> Tuple[np.ndarray, List[Dict]]:
        """Flip image and adjust annotations accordingly"""
        flip_code = 1 if horizontal else 0
        flipped_img = cv2.flip(image, flip_code)
        
        h_img, w_img = image.shape[:2]
        new_annotations = []
        
        for ann in annotations:
            new_ann = ann.copy()
            if horizontal:
                # x is normalized [0, 1]
                # New X = 1.0 - (x + width)
                new_ann['x'] = 1.0 - (ann['x'] + ann['width'])
            else:
                # New Y = 1.0 - (y + height)
                new_ann['y'] = 1.0 - (ann['y'] + ann['height'])
            
            # Handle polygons if they exist
            if 'points' in ann:
                new_points = []
                for pt in ann['points']:
                    if horizontal:
                        new_points.append({'x': 1.0 - pt['x'], 'y': pt['y']})
                    else:
                        new_points.append({'x': pt['x'], 'y': 1.0 - pt['y']})
                new_ann['points'] = new_points
                
            new_annotations.append(new_ann)
            
        return flipped_img, new_annotations

    @staticmethod
    def apply_gaussian_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add random Gaussian noise to the image"""
        noise = np.random.normal(0, intensity * 255, image.shape).astype(np.int16)
        noisy_img = image.astype(np.int16) + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_brightness_contrast(image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast"""
        # brightess: 0.5 to 1.5, contrast: 0.5 to 1.5
        return cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1.0) * 127)

    def generate_augmented_batch(
        self, 
        image_path: str, 
        annotations: List[Dict], 
        output_dir: Path,
        prefix: str,
        multiplier: int = 3
    ) -> List[Tuple[str, List[Dict]]]:
        """
        Generate multiple augmented versions of a single image.
        Returns list of (new_filepath, new_annotations)
        """
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        results = []
        
        # Strategies
        strategies = [
            ('horiz_flip', lambda i, a: self.apply_flip(i, a, True)),
            ('grayscale', lambda i, a: (self.apply_grayscale(i), a)),
            ('noise', lambda i, a: (self.apply_gaussian_noise(i, 0.05), a)),
            ('blur', lambda i, a: (self.apply_blur(i, 3), a)),
            ('bright', lambda i, a: (self.apply_brightness_contrast(i, 1.2, 1.1), a)),
            ('dark', lambda i, a: (self.apply_brightness_contrast(i, 0.8, 0.9), a)),
        ]
        
        # Pick random strategies based on multiplier
        selected = random.sample(strategies, min(multiplier, len(strategies)))
        
        for name, func in selected:
            aug_img, aug_ann = func(img, annotations)
            
            # Save new image
            filename = f"{prefix}_{name}.jpg"
            new_path = output_dir / filename
            cv2.imwrite(str(new_path), aug_img)
            
            results.append((str(new_path), aug_ann))
            
        return results

# Singleton
augmentor = DataAugmentor()
