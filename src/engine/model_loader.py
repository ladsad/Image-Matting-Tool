"""Model downloading and loading utilities."""

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from ..utils.paths import get_cache_dir, get_models_dir


# Model registry with URLs and metadata
# Using Hugging Face for reliable CDN hosting
MODEL_REGISTRY = {
    "modnet": {
        "name": "MODNet (Fast)",
        "description": "Trimap-free portrait matting - fast, good for portraits",
        "filename": "modnet.onnx",
        "url": "https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx",
        "size_mb": 25,
        "sha256": None,
        "requires_trimap": False,
        "quality": "standard",
        "input_format": "rgb",  # Model expects RGB normalized input
    },
    "modnet_photographic": {
        "name": "MODNet Photographic",
        "description": "MODNet finetuned for photographic portraits",
        "filename": "modnet_photographic_portrait_matting.onnx",
        "url": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0.0/modnet_photographic_portrait_matting.onnx",
        "size_mb": 25,
        "sha256": None,
        "requires_trimap": False,
        "quality": "high",
        "input_format": "rgb",
    },
    "vitmatte": {
        "name": "ViTMatte (Ultra)",
        "description": "SOTA Quality (ViTMatte) - Slow but precise",
        "filename": "vitmatte_base_composition_1k.onnx",
        "url": "https://huggingface.co/Xenova/vitmatte-base-composition-1k/resolve/main/onnx/model.onnx",
        "size_mb": 387,
        "sha256": None,
        "requires_trimap": True, # Handled internally by auto-trimap
        "quality": "ultra",
        "input_format": "rgb",
    },
}

class ModelLoader:
    """
    Handles model downloading, caching, and loading.
    
    Models are downloaded to the user's cache directory on first use
    and reused for subsequent runs.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Optional custom cache directory.
                      Defaults to user cache directory.
        """
        self.cache_dir = cache_dir or get_cache_dir() / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_models(self) -> dict:
        """Get list of available models with their metadata."""
        return MODEL_REGISTRY.copy()
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is downloaded and available."""
        if model_name not in MODEL_REGISTRY:
            return False
        
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Get the path to a model file.
        
        Args:
            model_name: Name of the model from MODEL_REGISTRY.
        
        Returns:
            Path to the model file.
        
        Raises:
            KeyError: If model_name is not in the registry.
        """
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
        return self.cache_dir / MODEL_REGISTRY[model_name]["filename"]
    
    def ensure_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Path:
        """
        Ensure a model is available, downloading if necessary.
        
        Args:
            model_name: Name of the model to ensure availability.
            progress_callback: Optional callback(progress: float, status: str)
                               where progress is 0.0 to 1.0.
        
        Returns:
            Path to the model file.
        
        Raises:
            KeyError: If model_name is not in the registry.
            RuntimeError: If download fails.
        """
        model_path = self.get_model_path(model_name)
        
        if model_path.exists():
            if progress_callback:
                progress_callback(1.0, "Model already downloaded")
            return model_path
        
        # Download the model
        model_info = MODEL_REGISTRY[model_name]
        url = model_info.get("url")
        
        if not url:
             if progress_callback:
                 progress_callback(0.0, f"Model {model_name} not found and no URL provided. Please install manually.")
             raise RuntimeError(f"Model {model_name} has no download URL and is not present locally.")
        
        if progress_callback:
            progress_callback(0.0, f"Downloading {model_info['name']}...")
        
        try:
            self._download_file(url, model_path, model_info.get("size_mb", 10), progress_callback)
        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")
        
        # Verify hash if provided
        if model_info.get("sha256"):
            if not self._verify_hash(model_path, model_info["sha256"]):
                model_path.unlink()
                raise RuntimeError("Model file hash verification failed")
        
        if progress_callback:
            progress_callback(1.0, "Download complete")
        
        return model_path
    
    def _download_file(
        self,
        url: str,
        dest_path: Path,
        expected_size_mb: float,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Download a file with progress tracking."""
        # Create a request with a user agent to avoid blocks
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        try:
            with urlopen(request, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                if total_size == 0:
                    total_size = int(expected_size_mb * 1024 * 1024)
                
                downloaded = 0
                chunk_size = 8192
                
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = downloaded / total_size
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            progress_callback(
                                progress,
                                f"Downloading: {mb_downloaded:.1f} / {mb_total:.1f} MB"
                            )
        
        except URLError as e:
            raise RuntimeError(f"Network error: {e}")
    
    def _verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file SHA256 hash."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest() == expected_hash
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cached model files.
        
        Args:
            model_name: Specific model to clear. If None, clears all models.
        """
        if model_name:
            model_path = self.get_model_path(model_name)
            if model_path.exists():
                model_path.unlink()
        else:
            for name in MODEL_REGISTRY:
                model_path = self.cache_dir / MODEL_REGISTRY[name]["filename"]
                if model_path.exists():
                    model_path.unlink()
