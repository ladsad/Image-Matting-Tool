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
        
        # RVM recurrent states (for video matting model)
        self._rvm_rec = None
        self._rvm_downsample_ratio = 0.25
        
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
        
        # Get input info
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        
        # Log which provider is being used
        active_provider = self.session.get_providers()[0]
        print(f"Matting engine initialized with {active_provider}")
    
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
        target_size = QUALITY_CONFIGS[quality]["resolution"]
        
        # Check if using RVM model
        is_rvm = self.model_info.get("recurrent", False) if self.model_info else False
        
        start_time = time.perf_counter()
        
        if is_rvm:
            # RVM-specific processing
            alpha = self._process_rvm(image, target_size)
        else:
            # MODNet/standard processing
            input_tensor, original_size, processed_size = self._preprocess(image, target_size)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            alpha = self._postprocess(outputs[0], original_size)
        
        inference_time = time.perf_counter() - start_time
        
        if return_timing:
            return alpha, inference_time
        return alpha
    
    def _process_rvm(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Process image using RVM (Robust Video Matting) model.
        
        RVM expects: src (NCHW RGB normalized), r1i, r2i, r3i, r4i (recurrent states), downsample_ratio
        Returns: fgr (foreground), pha (alpha), r1o, r2o, r3o, r4o (recurrent outputs)
        """
        original_h, original_w = image.shape[:2]
        
        # Resize to target while maintaining aspect ratio
        scale = target_size / max(original_h, original_w)
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)
        # Ensure divisible by 16 for RVM
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16
        new_h = max(new_h, 16)
        new_w = max(new_w, 16)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] and convert to NCHW
        src = resized.astype(np.float32) / 255.0
        src = np.transpose(src, (2, 0, 1))
        src = np.expand_dims(src, 0)
        
        # Get input names from model to understand expected format
        input_names = [inp.name for inp in self.session.get_inputs()]
        
        # RVM model has inputs: src, r1i, r2i, r3i, r4i, downsample_ratio
        # Initialize recurrent states based on actual required shapes
        # For single image processing, use zero-initialized recurrent states
        
        # Build inputs dict by matching input names
        inputs = {}
        
        for inp in self.session.get_inputs():
            name = inp.name
            
            if name == "src":
                inputs[name] = src
            elif name == "downsample_ratio":
                inputs[name] = np.array([self._rvm_downsample_ratio], dtype=np.float32)
            elif name.startswith("r") and name.endswith("i"):
                # Recurrent state input (r1i, r2i, r3i, r4i)
                # Shape is typically [batch, channels, height, width]
                # Channels vary between MobileNet and ResNet backbones
                
                # Default to MobileNet counts
                channels_map = {"r1i": 16, "r2i": 20, "r3i": 40, "r4i": 64}
                
                # ResNet50 counts (verified from model inspection)
                # Note: These differ from standard ResNet block widths
                if "resnet" in self.model_name.lower() or "rvm_resnet" in self.model_name.lower():
                     channels_map = {"r1i": 16, "r2i": 32, "r3i": 64, "r4i": 128}
                
                channels = channels_map.get(name, 16)
                
                # Height/width scale down based on level
                scale_map = {"r1i": 1, "r2i": 2, "r3i": 4, "r4i": 8}
                scale_factor = scale_map.get(name, 1)
                
                rec_h = new_h // scale_factor
                rec_w = new_w // scale_factor
                
                inputs[name] = np.zeros((1, channels, rec_h, rec_w), dtype=np.float32)
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Get output names to find alpha (pha)
        output_names = [out.name for out in self.session.get_outputs()]
        
        # Find alpha in outputs
        alpha = None
        for i, out_name in enumerate(output_names):
            if out_name == "pha":
                alpha = outputs[i][0, 0]  # (1, 1, H, W) -> (H, W)
                break
        
        if alpha is None:
            # Fallback: assume second output is alpha
            if len(outputs) >= 2:
                alpha = outputs[1][0, 0]
            else:
                alpha = outputs[0][0, 0]
        
        # Clip and convert
        alpha = np.clip(alpha, 0, 1)
        
        # Resize back to original
        alpha = cv2.resize(alpha, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to 0-255
        alpha = (alpha * 255).astype(np.uint8)
        
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
