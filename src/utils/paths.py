"""Path utilities for the Image Matting Tool."""

import os
import sys
from pathlib import Path


def get_app_dir() -> Path:
    """
    Get the application directory.
    
    When running as a PyInstaller bundle, returns the bundle directory.
    Otherwise, returns the project root directory.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running as script - go up from src/utils to project root
        return Path(__file__).parent.parent.parent


def get_models_dir() -> Path:
    """
    Get the models directory.
    
    Models are stored in the app directory when bundled,
    or in the project's models/ folder during development.
    """
    if getattr(sys, 'frozen', False):
        # When bundled, models are in the bundle
        return Path(sys._MEIPASS) / "models"
    else:
        return get_app_dir() / "models"


def get_cache_dir() -> Path:
    """
    Get the user cache directory for downloaded models and temporary files.
    
    Returns ~/.cache/matte on Unix-like systems,
    or %LOCALAPPDATA%/matte on Windows.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    
    cache_dir = base / "matte"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_config_dir() -> Path:
    """
    Get the user configuration directory.
    
    Returns ~/.config/matte on Unix-like systems,
    or %APPDATA%/matte on Windows.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    config_dir = base / "matte"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_logs_dir() -> Path:
    """Get the logs directory."""
    logs_dir = get_config_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
