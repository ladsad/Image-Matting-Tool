"""Core inference engine for image matting."""

from .matting import MattingEngine
from .model_loader import ModelLoader

__all__ = ["MattingEngine", "ModelLoader"]
