"""
Image utility functions for fingerprint computation, feature encoding, and display.

This module provides helper functions for image processing tasks that are used
across multiple tools and components.
"""

import os
import io
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image


def compute_image_fingerprint(image) -> str:
    """
    Return a stable hash for the uploaded image to detect reuse.
    
    Args:
        image: Image object (dict with 'path', str path, or PIL Image)
    
    Returns:
        SHA256 hash of the image content, or empty string on error
    """
    hasher = hashlib.sha256()
    try:
        if isinstance(image, dict) and 'path' in image:
            with open(image['path'], "rb") as f:
                hasher.update(f.read())
        elif isinstance(image, str) and os.path.exists(image):
            with open(image, "rb") as f:
                hasher.update(f.read())
        elif hasattr(image, "save"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            hasher.update(buf.getvalue())
        else:
            return ""
        return hasher.hexdigest()
    except Exception as e:
        print(f"Warning: failed to compute image fingerprint: {e}")
        return ""


def encode_image_features(image_path: str, features_dir: Path) -> str:
    """
    Compute a lightweight cached encoding for an image.
    This runs once per upload and is reused for all subsequent questions.
    
    Args:
        image_path: Path to the image file
        features_dir: Directory to store feature files
    
    Returns:
        Path to the saved feature file, or empty string on error
    """
    try:
        features_dir.mkdir(parents=True, exist_ok=True)
        feature_path = features_dir / (Path(image_path).stem + "_features.npy")
        if feature_path.exists():
            return str(feature_path)
        img = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pooled = np.concatenate([
            arr.mean(axis=(0, 1)),
            arr.std(axis=(0, 1)),
            arr.max(axis=(0, 1)),
            arr.min(axis=(0, 1))
        ])
        np.save(feature_path, pooled)
        return str(feature_path)
    except Exception as e:
        print(f"Warning: failed to encode image features for {image_path}: {e}")
        return ""


def load_image_for_display(image_path: str) -> Optional[Image.Image]:
    """
    Load an image for display purposes.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        PIL Image object, or None on error
    """
    try:
        if not os.path.exists(image_path):
            return None
        return Image.open(image_path)
    except Exception as e:
        print(f"Warning: failed to load image for display: {e}")
        return None

