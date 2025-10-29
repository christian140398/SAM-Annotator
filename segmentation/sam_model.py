"""
SAM Model Wrapper
Handles loading and managing the SAM model and predictor
"""
import os
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np


class SAMModel:
    """Wrapper for SAM model and predictor"""
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        """
        Initialize SAM model
        
        Args:
            checkpoint_path: Path to SAM checkpoint file
            model_type: Model type ("vit_b", "vit_l", "vit_h")
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        
        torch.set_grad_enabled(False)  # Disable gradients for inference
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.sam)
        self.current_image = None
        self.current_image_rgb = None
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
        """
        self.current_image = image.copy()
        # Convert BGR to RGB for SAM
        self.current_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.current_image_rgb)
    
    def predict_mask(self, points: list, labels: list, box: tuple = None) -> np.ndarray:
        """
        Predict mask from points
        
        Args:
            points: List of (x, y) tuples in image coordinates
            labels: List of 1 (positive) or 0 (negative) for each point
            box: Optional bounding box (xmin, ymin, xmax, ymax) to clip mask
            
        Returns:
            Boolean mask array, or None if prediction fails
        """
        if len(points) == 0:
            return None
        
        if len(points) != len(labels):
            raise ValueError("Points and labels must have same length")
        
        points_array = np.array(points, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        
        # Predict with SAM
        masks, scores, _ = self.predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=True
        )
        
        # Select best mask (highest score)
        best_mask = masks[np.argmax(scores)].astype(bool)
        
        # Clip to bounding box if provided
        if box is not None and self.current_image is not None:
            from .sam_utils import apply_clip_to_box
            H, W = self.current_image.shape[:2]
            best_mask = apply_clip_to_box(best_mask, box, H, W)
        
        # Only return mask if it has sufficient area
        if best_mask.sum() > 10:
            return best_mask
        else:
            return None
    
    def get_current_image(self) -> np.ndarray:
        """Get the current image (BGR format)"""
        return self.current_image

