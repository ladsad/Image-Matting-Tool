"""Core matting engine using ONNX Runtime for inference."""

import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .model_loader import ModelLoader
from ..utils.image_utils import apply_alpha_to_image


# Quality configuration presets
QUALITY_CONFIGS = {
    "standard": {
        "resolution": 256,
        "description": "Fast processing (~100ms)",
    },
    "high": {
        "resolution": 512,
        "description": "Balanced quality and speed (~200ms)",
    },
    "ultra": {
        "resolution": 1024,
        "description": "Best quality (~400ms)",
    },
}


class MattingEngine:
    """
    Main image matting engine using ONNX Runtime.
    
    This engine handles:
    - Model loading and session management
    - Image preprocessing and postprocessing
    - Alpha matte generation
    - Background application
    
    Example:
        engine = MattingEngine()
        alpha = engine.process_image("photo.jpg", quality="high")
        result = engine.apply_background(image, alpha, bg_color=(255, 255, 255))
    """
    
    def __init__(
        self,
        model_name: str = "modnet",
        use_gpu: bool = True,
        auto_download: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Initialize the matting engine.
        
        Args:
            model_name: Name of the model to use (from model registry).
            use_gpu: Whether to attempt GPU acceleration.
            auto_download: Whether to auto-download model if not present.
            progress_callback: Optional callback for download progress.
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )
        
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model_loader = ModelLoader()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.model_info: Optional[dict] = None
        
        # RVM recurrent states removed
        
        if auto_download:
            self._initialize_model(progress_callback)
    
    def _initialize_model(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Load the ONNX model and create inference session."""
        # Ensure model is available
        model_path = self.model_loader.ensure_model(
            self.model_name,
            progress_callback=progress_callback
        )
        
        # Get model info from registry
        self.model_info = self.model_loader.get_available_models().get(self.model_name, {})
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Select execution providers
        providers = []
        if self.use_gpu:
            # Try CUDA first, then DirectML for Windows
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            elif "DmlExecutionProvider" in ort.get_available_providers():
                providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")  # Fallback
        
        # Create session
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # For ViTMatte, we need an auxiliary MODNet model for trimap generation
        if self.model_name == "vitmatte":
            print("Initializing auxiliary MODNet for trimap generation...")
            modnet_path = self.model_loader.ensure_model("modnet")
            self.aux_session = ort.InferenceSession(
                str(modnet_path),
                sess_options=sess_options,
                providers=providers
            )
        else:
            self.aux_session = None
        
        # Get input info
        try:
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
        except Exception:
            # ViTMatte might be complex, skip single input check
            self.input_name = None
            self.input_shape = None
        
        # Log which provider is being used
        active_provider = self.session.get_providers()[0]
        print(f"Matting engine initialized with {active_provider}")

    def _generate_trimap(self, alpha: np.ndarray, convert_to_onehot: bool = True) -> np.ndarray:
        """
        Generate a trimap from coarse alpha.
        0=BG, 128=Unknown, 255=FG
        """
        if alpha.max() <= 1.0:
            alpha = (alpha * 255).astype(np.uint8)
        else:
            alpha = alpha.astype(np.uint8)
            
        # Erosion and Dilation to find unknown region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Thresholds
        fg_thresh = 240
        bg_thresh = 15
        
        # Initial Guess
        trimap = np.zeros_like(alpha)
        trimap[alpha >= fg_thresh] = 255 # FG
        trimap[alpha <= bg_thresh] = 0   # BG
        trimap[(alpha > bg_thresh) & (alpha < fg_thresh)] = 128 # Unknown
        
        # Dilate unknown region
        # Identify unknown
        unknown = (trimap == 128).astype(np.uint8)
        # Dilate unknown to cover edges better
        unknown = cv2.dilate(unknown, kernel, iterations=10)
        
        trimap[unknown == 1] = 128
        
        if not convert_to_onehot:
            return trimap
            
        # Convert to One-Hot (3, H, W) or Channel Map
        # ViTMatte usually expects specific trimap format.
        # We'll return the 1-channel trimap and preprocess it in _process_vitmatte
        return trimap

    def _process_vitmatte(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Process image using ViTMatte with auto-generated trimap."""
        h, w = image.shape[:2]
        
        # 1. Run MODNet (Aux) to get coarse alpha
        # Preprocess for MODNet (Standard preprocess: RGB, Normalized, etc.)
        modnet_input, orig_mod, proc_mod = self._preprocess(image, target_size=512) 
        
        # Run MODNet
        mod_out = self.aux_session.run(None, {self.aux_session.get_inputs()[0].name: modnet_input})
        mod_alpha = self._postprocess(mod_out[0], orig_mod)
        
        # 2. Prepare Inputs for ViTMatte
        # Image: (1, 3, H, W). Normalized with 0.5 mean/std.
        # Trimap: (1, 1, H, W) or (1, 4, H, W)?
        
        # Resize image to target size (divisible by 32)
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        new_h = max(new_h, 32)
        new_w = max(new_w, 32)
        
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Resize coarse alpha to match
        mod_alpha_resized = cv2.resize(mod_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Generate Trimap (Single channel: 0, 128, 255)
        trimap = self._generate_trimap(mod_alpha_resized, convert_to_onehot=False)
        
        # Prepare Image Tensor: (1, 3, H, W)
        # ViTMatte Mean/Std = 0.5
        pixel_values = img_resized.astype(np.float32) / 255.0
        pixel_values = (pixel_values - 0.5) / 0.5
        pixel_values = np.transpose(pixel_values, (2, 0, 1))
        pixel_values = np.expand_dims(pixel_values, 0).astype(np.float32)
        
        # Prepare Trimap Tensor
        # Inspect model input expectations via session
        input_names = [inp.name for inp in self.session.get_inputs()]
        
        inputs = {}
        if "pixel_values" in input_names:
            inputs["pixel_values"] = pixel_values
            
        # Handle Trimap
        # For ViTMatte ONNX, typically expects 4-channel (concat of image + trimap?) NO.
        # Usually 'trimap' or 'guidance'.
        
        trimap_name = "trimap" # Default guess
        for name in input_names:
             if "trimap" in name or "guidance" in name:
                 trimap_name = name
                 break
        
        # ViTMatte trimap processing:
        # 0 -> Background, 1 -> Unknown, 2 -> Foreground? Or 0-1 float?
        # HuggingFace Transformers logic:
        # It maps 0, 128, 255 to classes or keeps as float.
        # Let's try standard float trimap: 0.0, 0.5, 1.0
        
        trimap_float = trimap.astype(np.float32) / 255.0 # (H, W) in [0, 1]
        
        # If input shape expects 4 dims (N, C, H, W)
        trimap_tensor = np.expand_dims(trimap_float, 0) # (1, H, W)
        trimap_tensor = np.expand_dims(trimap_tensor, 0) # (1, 1, H, W)
        
        if trimap_name in input_names:
            inputs[trimap_name] = trimap_tensor.astype(np.float32)
            
        outputs = self.session.run(None, inputs)
        
        # Output is usually 'alphas' or first output
        alpha_out = outputs[0] # (1, 1, H, W)
        
        alpha_out = alpha_out[0, 0, :, :]
        
        # Resize back
        alpha_final = cv2.resize(alpha_out, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha_final = np.clip(alpha_final * 255, 0, 255).astype(np.uint8)
        
        return alpha_final

    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        quality: str = "high",
        return_timing: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Process an image and return the alpha matte.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError("Could not load image")
            
        original_h, original_w = image.shape[:2]
        
        # Handle quality settings
        target_sizes = {
            "standard": 512,
            "high": 1024,
            "ultra": 2048
        }
        target_size = target_sizes.get(quality, 1024)
        
        start_time = time.perf_counter()
        
        if self.model_name == "vitmatte":
             alpha = self._process_vitmatte(image, target_size)
        else:
            # MODNet/standard processing
            input_tensor, original_size, processed_size = self._preprocess(image, target_size)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            alpha = self._postprocess(outputs[0], original_size)
        
        process_time = time.perf_counter() - start_time
        
        if return_timing:
            return alpha, process_time
        return alpha
    
    def get_available_providers(self) -> List[str]:
        """Get list of available ONNX execution providers."""
        if ort is None:
            return []
        return ort.get_available_providers()
    
    def get_active_provider(self) -> str:
        """Get the currently active execution provider."""
        if self.session is None:
            return "None"
        return self.session.get_providers()[0]
    
    def _preprocess(
        self,
        image: np.ndarray,
        target_size: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess image for model input.
        
        MODNet expects input dimensions divisible by 32. We resize based on
        a reference size while maintaining aspect ratio.
        
        Args:
            image: Input image (H, W, 3) in RGB format.
            target_size: Target reference size for the longer dimension.
        
        Returns:
            Tuple of (preprocessed tensor, original size (W,H), processed size (W,H)).
        """
        original_size = (image.shape[1], image.shape[0])  # (W, H)
        h, w = image.shape[:2]
        
        # Calculate scale based on reference size
        # Use the longer dimension as reference
        ref_size = target_size
        if max(h, w) < ref_size or min(h, w) > ref_size:
            if w >= h:
                im_rh = ref_size
                im_rw = int(w / h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(h / w * ref_size)
        else:
            im_rh = h
            im_rw = w
        
        # Make dimensions divisible by 32
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        
        # Ensure minimum size
        im_rw = max(im_rw, 32)
        im_rh = max(im_rh, 32)
        
        # Resize image
        image = cv2.resize(image, (im_rw, im_rh), interpolation=cv2.INTER_AREA)
        processed_size = (im_rw, im_rh)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean/std (MODNet uses this normalization)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to NCHW format (batch, channels, height, width)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image, original_size, processed_size
    
    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess model output to alpha matte.
        
        Args:
            output: Raw model output.
            original_size: Original image size (W, H) to resize back to.
        
        Returns:
            Alpha matte as (H, W) array with values 0-255.
        """
        # MODNet outputs alpha in [0, 1] range
        alpha = output[0, 0]  # Remove batch and channel dimensions
        
        # Ensure in [0, 1]
        alpha = np.clip(alpha, 0, 1)
        
        # Resize to original size
        alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to 0-255 range
        alpha = (alpha * 255).astype(np.uint8)
        
        return alpha
    
    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        quality: str = "high",
        return_timing: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Process an image and generate alpha matte.
        
        Args:
            image: Input image path or numpy array (RGB format).
            quality: Quality preset ("standard", "high", "ultra").
            return_timing: If True, also return inference time in seconds.
        
        Returns:
            Alpha matte as (H, W) array with values 0-255.
            If return_timing is True, returns (alpha, time_seconds).
        
        Raises:
            ValueError: If quality preset is invalid.
            RuntimeError: If model is not loaded.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        if quality not in QUALITY_CONFIGS:
            raise ValueError(f"Invalid quality: {quality}. Options: {list(QUALITY_CONFIGS.keys())}")
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from ..utils.image_utils import load_image
            image = load_image(image)
        
        # Store original for compositing
        original_rgb = image.copy()
        
        # Get target resolution
        # Get target resolution
        target_size = QUALITY_CONFIGS[quality]["resolution"]
        
        start_time = time.perf_counter()
        
        if self.model_name == "vitmatte":
             alpha = self._process_vitmatte(image, target_size)
        else:
            # MODNet/standard processing
            input_tensor, original_size, processed_size = self._preprocess(image, target_size)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            alpha = self._postprocess(outputs[0], original_size)
        
        inference_time = time.perf_counter() - start_time
        
        if return_timing:
            return alpha, inference_time
        return alpha
    

    def process_with_background(
        self,
        image: Union[str, Path, np.ndarray],
        quality: str = "high",
        background: Optional[Union[Tuple[int, int, int], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Process image and apply background in one step.
        
        Args:
            image: Input image path or numpy array.
            quality: Quality preset.
            background: Background color (R, G, B) or image array.
                       None for transparent output.
        
        Returns:
            Processed image (RGBA if transparent, RGB otherwise).
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from ..utils.image_utils import load_image
            image = load_image(image)
        
        # Get alpha matte
        alpha = self.process_image(image, quality=quality)
        
        # Apply background
        return apply_alpha_to_image(image, alpha, background)
    
    def batch_process(
        self,
        image_paths: List[Union[str, Path]],
        quality: str = "high",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths to process.
            quality: Quality preset for all images.
            progress_callback: Optional callback(current, total, status).
        
        Returns:
            List of (alpha, original_image) tuples.
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, total, f"Processing {Path(path).name}")
            
            try:
                from ..utils.image_utils import load_image
                image = load_image(path)
                alpha = self.process_image(image, quality=quality)
                results.append((alpha, image))
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append((None, None))
        
        if progress_callback:
            progress_callback(total, total, "Complete")
        
        return results
    
    @staticmethod
    def get_quality_options() -> dict:
        """Get available quality presets and their descriptions."""
        return QUALITY_CONFIGS.copy()
    
    @staticmethod
    def recommend_quality(image_size_mb: float) -> str:
        """
        Recommend quality preset based on image size.
        
        Args:
            image_size_mb: Image file size in megabytes.
        
        Returns:
            Recommended quality preset name.
        """
        if image_size_mb < 2:
            return "standard"
        elif image_size_mb < 10:
            return "high"
        else:
            return "ultra"
