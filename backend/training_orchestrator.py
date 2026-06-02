"""
Training Orchestrator
Manages automatic training and fine-tuning workflows
"""

import logging
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Dict, Optional
import random

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Manages automatic training and fine-tuning workflows"""
    
    # Training milestones
    INITIAL_TRAINING_THRESHOLD = 10  # Start initial training at 10 verified images
    FINE_TUNE_THRESHOLD = 10  # Start fine-tuning after 10 corrections
    
    # Hyperparameters
    INITIAL_EPOCHS = 25  # Initial training: 20-30 epochs
    FINE_TUNE_EPOCHS = 7  # Fine-tuning: 5-10 epochs
    
    @classmethod
    def check_auto_train_eligibility(cls, project, db: Session) -> Dict:
        """
        Check if project is eligible for automatic initial training
        
        Returns:
            dict with 'eligible', 'verified_count', and 'message'
        """
        from database.models import Annotation, Image
        from sqlalchemy import func, distinct
        
        # Count verified images directly using the verification_status field
        verified_count = db.query(Image).filter(
            Image.project_id == project.id,
            Image.verification_status == "verified"
        ).count()
        
        # Check if we have significantly more images than the last training session
        last_trained_count = project.label_schema.get('last_trained_image_count', 0)
        
        # ONLY trigger auto-training for the VERY FIRST model
        if last_trained_count == 0:
            eligible = verified_count >= cls.INITIAL_TRAINING_THRESHOLD
            message = f'Initial training {"ready" if eligible else "pending"}: {verified_count}/{cls.INITIAL_TRAINING_THRESHOLD} verified images'
        else:
            # Once we have a model, we don't trigger re-training based on image count alone
            eligible = False
            message = 'Initial model exists. Waiting for corrections for fine-tuning.'
        
        return {
            'eligible': eligible,
            'has_custom_model': last_trained_count > 0 or project.label_schema.get('latest_model') is not None,
            'verified_count': verified_count,
            'last_trained_count': last_trained_count,
            'threshold': cls.INITIAL_TRAINING_THRESHOLD,
            'message': message
        }
    
    @classmethod
    def check_fine_tune_eligibility(cls, project, db: Session) -> Dict:
        """
        Check if project is eligible for fine-tuning based on correction count
        
        Note: This tracks corrections made to AI annotations.
        A correction is when a user saves over an existing AI-generated annotation.
        """
        # Check if we have a custom model first
        has_custom_model = project.label_schema.get('latest_model') is not None
        
        if not has_custom_model:
            return {
                'eligible': False,
                'correction_count': 0,
                'message': 'No custom model trained yet'
            }
        
        # Get correction count from project metadata
        correction_count = project.label_schema.get('correction_count', 0)
        last_trained_correction_count = project.label_schema.get('last_trained_correction_count', 0)
        new_corrections = correction_count - last_trained_correction_count
        
        # Eligible if we have at least 10 new corrections since last train
        # (This avoids retraining for just 5 images even if they are corrections)
        eligible = new_corrections >= 10
        
        return {
            'eligible': eligible,
            'correction_count': correction_count,
            'last_trained_correction_count': last_trained_correction_count,
            'new_corrections': new_corrections,
            'threshold': 10,
            'message': f'Fine-tuning {"ready" if eligible else "pending"}: {new_corrections}/10 new corrections'
        }
    
    @classmethod
    def get_training_hyperparameters(cls, is_initial: bool = True, is_segmentation: bool = False) -> Dict:
        """
        Get hyperparameters for training
        
        Args:
            is_initial: Whether this is initial training or fine-tuning
            is_segmentation: Whether the project uses segmentation
        
        Returns:
            dict with training parameters
        """
        if is_initial:
            return {
                'epochs': cls.INITIAL_EPOCHS,
                'batch': 16,
                'imgsz': 640,
                'lr0': 0.01,  # Standard learning rate for initial training
                'augment_multiplier': 10,  # Expand from 10 images to 100
                'freeze_backbone': 10,  # Freeze backbone for first 10 epochs
            }
        else:
            # Fine-tuning parameters
            return {
                'epochs': cls.FINE_TUNE_EPOCHS,
                'batch': 8,
                'imgsz': 640,
                'lr0': 0.001,  # Lower learning rate for fine-tuning (10x reduction)
                'augment_multiplier': 10,  # Same augmentation
                'freeze_backbone': 0,  # Don't freeze for fine-tuning
            }
    
    @classmethod
    def increment_correction_count(cls, project, db: Session) -> int:
        """
        Increment the correction count for a project
        
        Returns:
            new correction count
        """
        from sqlalchemy.orm.attributes import flag_modified
        
        schema = project.label_schema.copy()
        correction_count = schema.get('correction_count', 0) + 1
        schema['correction_count'] = correction_count
        
        project.label_schema = schema
        flag_modified(project, "label_schema")
        db.commit()
        
        logger.info(f"Project {project.name}: Correction count incremented to {correction_count}")
        
        return correction_count
    
    @classmethod
    def reset_correction_count(cls, project, db: Session):
        """Reset correction count after fine-tuning completes"""
        from sqlalchemy.orm.attributes import flag_modified
        
        schema = project.label_schema.copy()
        schema['correction_count'] = 0
        
        project.label_schema = schema
        flag_modified(project, "label_schema")
        db.commit()
        
        logger.info(f"Project {project.name}: Correction count reset")
    
    @classmethod
    def randomize_dataset(cls, image_paths: list) -> list:
        """
        Randomize dataset order for better training
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            Shuffled list
        """
        randomized = image_paths.copy()
        random.shuffle(randomized)
        return randomized
    
    @classmethod
    def split_dataset(cls, image_paths: list, train_ratio: float = 0.8, val_ratio: float = 0.15) -> Dict:
        """
        Split dataset into train/val/test sets
        
        Args:
            image_paths: List of image paths
            train_ratio: Ratio for training set (default 0.8)
            val_ratio: Ratio for validation set (default 0.15, test gets remainder)
        
        Returns:
            dict with 'train', 'val', 'test' lists
        """
        # Randomize first
        shuffled = cls.randomize_dataset(image_paths)
        
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        return {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }
