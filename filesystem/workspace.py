import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
from dataclasses import dataclass
from enum import Enum
import json

class WorkspaceError(Exception):
    """Workspace operation errors"""
    pass

@dataclass
class ProjectWorkspace:
    """Manages project-specific directory structure"""
    base_path: Path
    project_name: str
    
    @property
    def project_dir(self) -> Path:
        return self.base_path / "projects" / self.project_name
    
    @property
    def raw_videos_dir(self) -> Path:
        return self.project_dir / "raw_videos"
    
    @property
    def images_dir(self) -> Path:
        return self.project_dir / "images"
    
    @property
    def annotations_dir(self) -> Path:
        return self.project_dir / "annotations"
    
    @property
    def exports_dir(self) -> Path:
        return self.project_dir / "exports"
    
    def create(self) -> None:
        """Create project directory structure"""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.raw_videos_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
    
    def get_next_annotation_version(self, image_filename: str) -> int:
        """Get next version number for image annotations"""
        ann_dir = self.annotations_dir / Path(image_filename).stem
        if not ann_dir.exists():
            return 1
        
        versions = [int(d.name[1:]) for d in ann_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        return max(versions, default=0) + 1
    
    def save_annotation(self, image_filename: str, version: int, data: Dict) -> Path:
        """Save annotation to versioned directory"""
        ann_dir = self.annotations_dir / Path(image_filename).stem / f"v{version}"
        ann_dir.mkdir(parents=True, exist_ok=True)
        
        ann_file = ann_dir / "annotation.json"
        with open(ann_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return ann_file
    
    def get_annotation_path(self, image_filename: str, version: Optional[int] = None) -> Optional[Path]:
        """Get path to annotation file"""
        base_dir = self.annotations_dir / Path(image_filename).stem
        
        if not base_dir.exists():
            return None
        
        if version is None:
            # Get latest version
            versions = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
            if not versions:
                return None
            version_dirs = sorted(versions, key=lambda d: int(d.name[1:]))
            ann_dir = version_dirs[-1]
        else:
            ann_dir = base_dir / f"v{version}"
        
        ann_file = ann_dir / "annotation.json"
        return ann_file if ann_file.exists() else None

class VideoProcessor:
    """Handles video frame extraction"""
    
    @staticmethod
    def estimate_total_frames(video_path: Path, target_fps: float) -> Tuple[int, float]:
        """Estimate total frames for extraction"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise WorkspaceError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if video_fps <= 0:
            raise WorkspaceError("Invalid video FPS")
        
        # Calculate frame interval based on target FPS
        frame_interval = max(1, int(video_fps / target_fps))
        estimated_frames = total_frames // frame_interval
        
        return estimated_frames, video_fps
    
    @staticmethod
    def extract_frames(
        video_path: Path,
        output_dir: Path,
        target_fps: float,
        progress_callback=None,
        video_hash: str = None
    ) -> List[Path]:
        """Extract frames from video using randomized/stratified sampling for diversity"""
        import random
        import numpy as np
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate base frame interval
        frame_interval = max(1, int(video_fps / target_fps))
        
        # Calculate expected number of frames
        expected_frames = total_frames // frame_interval
        
        # Generate randomized frame indices with stratified sampling
        # This ensures coverage across entire video while adding diversity
        frame_indices = []
        for i in range(expected_frames):
            # Base position for this segment
            base_position = i * frame_interval
            
            # Add random jitter (±30% of interval) for diversity
            jitter_range = int(frame_interval * 0.3)
            jitter = random.randint(-jitter_range, jitter_range)
            
            # Ensure we stay within valid range
            frame_idx = max(0, min(total_frames - 1, base_position + jitter))
            frame_indices.append(frame_idx)
        
        # Sort indices for efficient sequential read with seeks
        frame_indices = sorted(set(frame_indices))  # Remove any duplicates and sort
        
        # Use video hash or stem for naming prefix
        prefix = video_hash[:12] if video_hash else video_path.stem
        
        frame_paths = []
        saved_count = 0
        current_frame = 0
        
        for target_frame in frame_indices:
            # Seek to target frame if necessary
            if target_frame != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame = target_frame
            
            ret, frame = cap.read()
            if not ret:
                current_frame += 1
                continue
            
            # Include FPS in filename to distinguish different extraction settings
            filename = f"{prefix}_{target_fps}fps_frame{saved_count:06d}.jpg"
            output_path = output_dir / filename
            
            # Check if file already exists to avoid redundant Disk I/O
            if not output_path.exists():
                # Save as JPEG with quality 95
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            frame_paths.append(output_path)
            saved_count += 1
            current_frame += 1
            
            if progress_callback:
                progress_callback(saved_count)
        
        cap.release()
        return frame_paths

class ImageImporter:
    """Handles image import with deduplication"""
    
    @staticmethod
    def calculate_image_hash(image_path: Path) -> str:
        """Calculate SHA256 hash of image for deduplication"""
        hasher = hashlib.sha256()
        with open(image_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions using OpenCV"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise WorkspaceError(f"Cannot read image: {image_path}")
        return img.shape[1], img.shape[0]  # width, height
    
    @classmethod
    def import_image(cls, source_path: Path, dest_dir: Path) -> Tuple[Path, str]:
        """Import single image with deduplication"""
        if not source_path.exists():
            raise WorkspaceError(f"Source image not found: {source_path}")
        
        # Calculate hash for deduplication
        image_hash = cls.calculate_image_hash(source_path)
        dest_filename = f"{image_hash[:16]}_{source_path.name}"
        dest_path = dest_dir / dest_filename
        
        # Copy if doesn't exist
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
        
        return dest_path, image_hash
    
    @classmethod
    def import_folder(cls, source_dir: Path, dest_dir: Path) -> List[Tuple[Path, str]]:
        """Import all images from folder"""
        if not source_dir.exists() or not source_dir.is_dir():
            raise WorkspaceError(f"Source directory not found: {source_dir}")
        
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for img_path in source_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                try:
                    dest_path, img_hash = cls.import_image(img_path, dest_dir)
                    results.append((dest_path, img_hash))
                except Exception as e:
                    print(f"Error importing {img_path}: {e}")
        
        return results
