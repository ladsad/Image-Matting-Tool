"""Configuration management for the Image Matting Tool."""

import json
from pathlib import Path
from typing import Any, Optional

from .paths import get_config_dir


# Default configuration values
DEFAULT_CONFIG = {
    "quality": "high",
    "background": "transparent",
    "custom_bg_color": [255, 255, 255],
    "use_gpu": True,
    "last_input_folder": "",
    "last_output_folder": "",
    "window_position": None,
    "window_size": None,
    "batch_workers": 2,
    "auto_save_settings": True,
    "output_format": "png",
    "jpeg_quality": 95,
}


class ConfigManager:
    """
    Manage user preferences with JSON persistence.
    
    Configuration is stored in the user's config directory
    and persists between application sessions.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional custom path for the config file.
                        Defaults to user config directory.
        """
        self.config_path = config_path or (get_config_dir() / "settings.json")
        self._config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from file, merging with defaults."""
        config = DEFAULT_CONFIG.copy()
        
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                    config.update(saved_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config: {e}")
        
        return config
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key to retrieve.
            default: Default value if key doesn't exist.
        
        Returns:
            The configuration value or default.
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key to set.
            value: Value to store.
            auto_save: Whether to save immediately (if auto_save_settings is enabled).
        """
        self._config[key] = value
        if auto_save and self._config.get("auto_save_settings", True):
            self.save()
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset configuration to defaults.
        
        Args:
            key: Specific key to reset. If None, resets all settings.
        """
        if key is None:
            self._config = DEFAULT_CONFIG.copy()
        elif key in DEFAULT_CONFIG:
            self._config[key] = DEFAULT_CONFIG[key]
        self.save()
    
    def get_all(self) -> dict:
        """Get all configuration values."""
        return self._config.copy()
    
    def update(self, updates: dict) -> None:
        """Update multiple configuration values at once."""
        self._config.update(updates)
        if self._config.get("auto_save_settings", True):
            self.save()
