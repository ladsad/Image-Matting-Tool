"""
Advanced image processing algorithms for matting improvements.
Includes preprocessing (to help the model) and postprocessing (to clean up the result).
"""

import cv2
import numpy as np
from typing import Optional, Tuple

def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to the image.
    Values < 1.0 darken the image, values > 1.0 brighten it.
    
    Args:
        image: Input image (H, W, C) or (H, W).
        gamma: Gamma value.
        
    Returns:
        Gamma corrected image.
    """
    if gamma == 1.0:
        return image

    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def apply_auto_contrast(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Converts to LAB color space, applies CLAHE to L channel, and converts back.
    
    Args:
        image: Input RGB image.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of grid for histogram equalization.
        
    Returns:
        Contrast enhanced image.
    """
    if image.ndim != 3:
        return image # TODO: Handle grayscale
        
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    
    # Merge and convert back
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final

def apply_denoising(image: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        image: Input image.
        strength: Sigma value for Gaussian kernel.
        
    Returns:
        Denoised image.
    """
    if strength <= 0:
        return image
        
    # Kernel size must be odd
    ksize = int(strength * 3)
    if ksize % 2 == 0:
        ksize += 1
        
    return cv2.GaussianBlur(image, (ksize, ksize), strength)

def refine_alpha_morphology(alpha: np.ndarray, kernel_size: int = 3, operation: str = "open-close") -> np.ndarray:
    """
    Apply morphological operations to clean up the alpha matte.
    
    Args:
        alpha: Input alpha matte (0-255).
        kernel_size: Size of the structuring element.
        operation: 'open-close' (remove noise then fill holes) or specific cv2 morph op.
        
    Returns:
        Cleaned alpha matte.
    """
    if kernel_size <= 1:
        return alpha
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == "open-close":
        # Remove small white noise (Erode -> Dilate)
        opened = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        # Fill small black holes (Dilate -> Erode)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return result
    else:
        return cv2.morphologyEx(alpha, getattr(cv2, operation), kernel)

def remove_islands(alpha: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    """
    Keep only the largest connected components.
    
    Args:
        alpha: Input alpha matte (0-255).
        min_area_ratio: Minimum area relative to the largest component to keep. 
                        Components smaller than this fraction of the max area will be removed.
        
    Returns:
        Alpha matte with small islands removed.
    """
    # Threshold to binary for analysis
    _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Connected components
    # connectivity=8 looks at diagonal neighbors too
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:
        return alpha
        
    # stats[:, 4] is area. Index 0 is background (usually label 0 is background if we threshold properly)
    # However, sometimes label 0 is just the first component. 
    # Usually connectedComponents treats 0 as background.
    
    # Get areas of all foreground components (skip label 0 which is background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if len(areas) == 0:
        return alpha
        
    max_area = np.max(areas)
    threshold_area = max_area * min_area_ratio
    
    # Identify labels to keep
    # We keep components that are large enough
    labels_to_keep = [i + 1 for i, area in enumerate(areas) if area >= threshold_area]
    
    # Create mask of kept labels
    mask = np.isin(labels, labels_to_keep).astype(np.uint8) * 255
    
    # Apply mask to original alpha
    # We want to keep the original alpha values where the mask is present, and 0 elsewhere
    # But we need to be careful not to zero out soft edges that might drift into 'background' label if we just use binary mask directly.
    # A standard approach is to use the mask as a hard clip, or use morphology to dilate the mask slightly before applying.
    # For now, let's just mask it.
    
    return cv2.bitwise_and(alpha, alpha, mask=mask)

def refine_edges_guided(image: np.ndarray, alpha: np.ndarray, radius: int = 15, eps: float = 1e-3) -> np.ndarray:
    """
    Refine alpha matte edges using the original image as a guide (Guided Filter).
    
    Args:
        image: Guide image (RGB).
        alpha: Input alpha matte.
        radius: Radius of the guided filter.
        eps: Regularization parameter (epsilon).
        
    Returns:
        Refined alpha matte.
    """
    # Normalize
    alpha_norm = alpha.astype(np.float32) / 255.0
    image_norm = image.astype(np.float32) / 255.0
    
    # Check if ximgproc is available (opencv-contrib)
    try:
        from cv2.ximgproc import guidedFilter
        refined = guidedFilter(image_norm, alpha_norm, radius, eps)
    except ImportError:
        # Fallback to a simpler edge preserving filter if guided filter is missing
        # Or notify user? For now, let's use a blurred version as a poor man's fallback or just return alpha
        # Actually, let's just return alpha if we can't do it, to avoid degrading.
        # But we can try to implement a fast guided filter using basic cv2 functions if we really want.
        # For this MVP, let's assume standard opencv might not have ximgproc unless installed with contrib.
        # We will try to use the Fast Guided Filter if we implement it, but for now return alpha.
        print("Warning: cv2.ximgproc not found. Guided Filter skipped.")
        return alpha
        
    # Denormalize
    return (refined * 255).clip(0, 255).astype(np.uint8)
