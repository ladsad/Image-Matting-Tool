"""Image loading and saving utilities."""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}


def load_image(
    path: Union[str, Path],
    max_size: Optional[int] = None,
    convert_rgb: bool = True
) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file.
        max_size: Optional maximum dimension. Image will be resized if larger.
        convert_rgb: If True, convert to RGB format (default for display).
    
    Returns:
        NumPy array of the image in RGB format (H, W, 3) or RGBA (H, W, 4).
    
    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the file format is not supported.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {suffix}. Supported: {SUPPORTED_FORMATS}")
    
    # Use PIL for loading to handle various formats consistently
    with Image.open(path) as img:
        # Convert to RGB/RGBA
        if img.mode == "RGBA":
            image = np.array(img)
        elif img.mode == "RGB":
            image = np.array(img)
        else:
            # Convert grayscale, palette, etc. to RGB
            image = np.array(img.convert("RGB"))
    
    # Resize if needed
    if max_size is not None:
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    format: Optional[str] = None,
    quality: int = 95
) -> Path:
    """
    Save an image to disk.
    
    Args:
        image: NumPy array of the image (RGB or RGBA format).
        path: Output path for the image.
        format: Optional format override. If None, determined from path extension.
        quality: JPEG quality (1-100). Only used for JPEG format.
    
    Returns:
        Path to the saved image.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format is None:
        format = path.suffix.lower().lstrip(".")
    
    # Convert RGBA to PIL Image
    if image.ndim == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image, mode="RGBA")
    elif image.ndim == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image, mode="RGB")
    elif image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Save with appropriate options
    save_kwargs = {}
    if format.lower() in ("jpg", "jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
        # JPEG doesn't support transparency, convert to RGB
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
    elif format.lower() == "png":
        save_kwargs["optimize"] = True
    elif format.lower() == "webp":
        save_kwargs["quality"] = quality
    
    pil_image.save(path, **save_kwargs)
    return path


def get_image_info(path: Union[str, Path]) -> dict:
    """
    Get information about an image file.
    
    Args:
        path: Path to the image file.
    
    Returns:
        Dictionary with image information (size, format, dimensions, etc.)
    """
    path = Path(path)
    
    with Image.open(path) as img:
        return {
            "path": str(path),
            "filename": path.name,
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": path.stat().st_size / (1024 * 1024),
        }


def resize_for_preview(
    image: np.ndarray,
    max_width: int = 400,
    max_height: int = 400
) -> np.ndarray:
    """
    Resize an image to fit within preview dimensions while maintaining aspect ratio.
    
    Args:
        image: Input image as NumPy array.
        max_width: Maximum width for preview.
        max_height: Maximum height for preview.
    
    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    
    # Calculate scale to fit within bounds
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """
    Convert a NumPy image array to bytes for display in GUI.
    
    Args:
        image: NumPy array of the image.
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Image as bytes.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image, mode="RGBA")
    elif image.ndim == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image, mode="RGB")
    else:
        pil_image = Image.fromarray(image, mode="L")
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return buffer.getvalue()


def apply_alpha_to_image(
    image: np.ndarray,
    alpha: np.ndarray,
    background: Optional[Union[Tuple[int, int, int], np.ndarray]] = None
) -> np.ndarray:
    """
    Apply an alpha matte to an image.
    
    Args:
        image: RGB image as NumPy array (H, W, 3).
        alpha: Alpha matte as NumPy array (H, W) with values 0-255.
        background: Either a tuple (R, G, B) for solid color,
                   an ndarray for background image, or None for transparent.
    
    Returns:
        RGBA image if background is None, otherwise RGB image.
    """
    h, w = image.shape[:2]
    
    # Ensure alpha is the right shape
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize alpha to 0-1
    if alpha.max() > 1:
        alpha_norm = alpha.astype(np.float32) / 255.0
    else:
        alpha_norm = alpha.astype(np.float32)
    
    if background is None:
        # Create RGBA image with transparency
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = (alpha_norm * 255).astype(np.uint8)
        return rgba
    
    elif isinstance(background, tuple):
        # Solid color background
        bg = np.full((h, w, 3), background, dtype=np.uint8)
    else:
        # Background image
        bg = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
        if bg.shape[2] == 4:
            bg = bg[:, :, :3]
    
    # Composite: result = fg * alpha + bg * (1 - alpha)
    alpha_3ch = alpha_norm[:, :, np.newaxis]
    result = (image.astype(np.float32) * alpha_3ch + 
              bg.astype(np.float32) * (1 - alpha_3ch))
    
    return result.astype(np.uint8)
