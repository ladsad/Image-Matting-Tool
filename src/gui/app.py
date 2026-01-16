"""Main application controller for the Image Matting Tool."""

import io
import threading
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import FreeSimpleGUI as sg
from PIL import Image

from ..engine import MattingEngine
from ..utils import ConfigManager, load_image, save_image
from ..utils.image_utils import (
    apply_alpha_to_image,
    get_image_info,
    image_to_bytes,
    resize_for_preview,
    SUPPORTED_FORMATS,
)
from .layouts import (
    create_main_layout,
    create_batch_layout,
    create_settings_layout,
    create_about_layout,
    get_theme,
    FONTS,
)


class MattingApp:
    """
    Main application controller.
    
    Handles:
    - Window creation and event loop
    - User interactions
    - Coordination between GUI and engine
    """
    
    PREVIEW_SIZE = (400, 400)
    
    def __init__(self):
        """Initialize the application."""
        self.config = ConfigManager()
        self.engine: Optional[MattingEngine] = None
        self.window: Optional[sg.Window] = None
        
        # Current state
        self.current_image: Optional[np.ndarray] = None
        self.current_alpha: Optional[np.ndarray] = None
        self.current_result: Optional[np.ndarray] = None
        self.current_path: Optional[Path] = None
        self.is_processing = False
        
        # Initialize theme
        get_theme()
    
    def _init_engine(self, progress_callback=None) -> Tuple[bool, Optional[str]]:
        """
        Initialize the matting engine.
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            self.engine = MattingEngine(
                model_name="modnet",
                use_gpu=self.config.get("use_gpu", True),
                auto_download=True,
                progress_callback=progress_callback,
            )
            return True, None
        except Exception as e:
            error_msg = (
                f"Failed to initialize matting engine:\n\n{e}\n\n"
                "Please check your internet connection for model download."
            )
            return False, error_msg
    
    def create_main_window(self) -> sg.Window:
        """Create and return the main application window."""
        layout = create_main_layout()
        
        window = sg.Window(
            "Image Matting Tool",
            layout,
            finalize=True,
            resizable=False,
            element_justification="center",
            margins=(20, 20),
            enable_close_attempted_event=True,
        )
        
        # Bind keyboard shortcuts
        window.bind("<Control-o>", "-KEY-CTRL-O-")  # Open file
        window.bind("<Control-O>", "-KEY-CTRL-O-")
        window.bind("<Control-s>", "-KEY-CTRL-S-")  # Save file
        window.bind("<Control-S>", "-KEY-CTRL-S-")
        window.bind("<Control-p>", "-KEY-CTRL-P-")  # Process
        window.bind("<Control-P>", "-KEY-CTRL-P-")
        window.bind("<Control-b>", "-KEY-CTRL-B-")  # Batch mode
        window.bind("<Control-B>", "-KEY-CTRL-B-")
        window.bind("<F5>", "-KEY-F5-")  # Process (alternative)
        
        # Enable drag-and-drop for the window
        window.TKroot.drop_target_register("DND_Files") if hasattr(window.TKroot, 'drop_target_register') else None
        
        # Restore window position if saved
        pos = self.config.get("window_position")
        if pos:
            window.move(pos[0], pos[1])
        
        return window
    
    def _get_quality_key(self, display_value: str) -> str:
        """Convert display quality name to config key."""
        mapping = {
            "Standard (Fast)": "standard",
            "High (Balanced)": "high",
            "Ultra (Best)": "ultra",
        }
        return mapping.get(display_value, "high")
    
    def _get_background_color(self, bg_option: str, custom_color: str) -> Optional[Tuple[int, int, int]]:
        """Get background color tuple from option."""
        if bg_option == "Transparent":
            return None
        elif bg_option == "White":
            return (255, 255, 255)
        elif bg_option == "Black":
            return (0, 0, 0)
        elif bg_option == "Custom Color..." and custom_color:
            # Parse hex color
            try:
                color = custom_color.lstrip("#")
                return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            except:
                return None
        return None
    
    def _update_preview(self, key: str, image: Optional[np.ndarray]) -> None:
        """Update a preview image in the window."""
        if image is None:
            self.window[key].update(data=None)
            return
        
        # Resize for preview
        preview = resize_for_preview(image, *self.PREVIEW_SIZE)
        
        # Convert to PNG bytes
        png_bytes = image_to_bytes(preview, format="PNG")
        
        self.window[key].update(data=png_bytes)
    
    def _load_image_file(self, filepath: str) -> bool:
        """
        Load an image file and update the UI.
        
        Args:
            filepath: Path to the image file.
        
        Returns:
            True if successful, False otherwise.
        """
        path = Path(filepath)
        
        # Validate file
        if not path.exists():
            sg.popup_error("File not found.", title="Error")
            return False
        
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            sg.popup_error(
                f"Unsupported format: {path.suffix}\n\n"
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
                title="Error"
            )
            return False
        
        try:
            # Load and display image
            self.current_image = load_image(path)
            self.current_path = path
            self.current_alpha = None
            self.current_result = None
            
            # Update preview
            self._update_preview("-INPUT-PREVIEW-", self.current_image)
            self._update_preview("-OUTPUT-PREVIEW-", None)
            
            # Update file info
            info = get_image_info(path)
            self.window["-FILE-INFO-"].update(
                f"{info['filename']} - {info['width']}x{info['height']} - "
                f"{info['file_size_mb']:.1f} MB"
            )
            self.window["-TIMING-INFO-"].update("")
            
            # Update drop hint
            if "-DROP-HINT-" in self.window.AllKeysDict:
                self.window["-DROP-HINT-"].update("Image loaded - select another to replace")
            
            # Enable process button
            self.window["-PROCESS-"].update(disabled=False)
            self.window["-SAVE-"].update(disabled=True)
            
            # Save folder preference
            self.config.set("last_input_folder", str(path.parent))
            
            return True
            
        except Exception as e:
            sg.popup_error(f"Failed to load image:\n\n{e}", title="Error")
            return False
    
    def _handle_file_select(self) -> None:
        """Handle image file selection via dialog."""
        # Get supported formats for file dialog
        formats = " ".join(f"*{ext}" for ext in SUPPORTED_FORMATS)
        
        filepath = sg.popup_get_file(
            "Select Image",
            file_types=[("Image Files", formats), ("All Files", "*.*")],
            initial_folder=self.config.get("last_input_folder", ""),
        )
        
        if filepath:
            self._load_image_file(filepath)
    
    def _handle_process(self) -> None:
        """Handle background removal processing."""
        if self.current_image is None:
            return
        
        if self.engine is None:
            self.window["-STATUS-"].update("Initializing engine...")
            self.window["-PROGRESS-"].update(visible=True)
            
            def progress_callback(progress, status):
                self.window.write_event_value("-ENGINE-PROGRESS-", (progress, status))
            
            # Initialize in background
            def init_thread():
                success, error_msg = self._init_engine(progress_callback)
                self.window.write_event_value("-ENGINE-READY-", (success, error_msg))
            
            threading.Thread(target=init_thread, daemon=True).start()
            return
        
        # Run processing
        self._run_processing()
    
    def _run_processing(self) -> None:
        """Execute the matting process."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.window["-PROCESS-"].update(disabled=True)
        self.window["-PROGRESS-"].update(visible=True, current_count=0)
        self.window["-STATUS-"].update("Processing...")
        
        # Get settings
        quality = self._get_quality_key(self.window["-QUALITY-"].get())
        bg_option = self.window["-BACKGROUND-"].get()
        custom_color = self.window["-CUSTOM-COLOR-"].get()
        background = self._get_background_color(bg_option, custom_color)
        
        def process_thread():
            try:
                # Process image
                alpha, timing = self.engine.process_image(
                    self.current_image,
                    quality=quality,
                    return_timing=True,
                )
                
                # Apply background
                result = apply_alpha_to_image(self.current_image, alpha, background)
                
                # Send result back to main thread
                self.window.write_event_value("-PROCESS-DONE-", (alpha, result, timing))
                
            except Exception as e:
                self.window.write_event_value("-PROCESS-ERROR-", str(e))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def _handle_process_complete(self, result: Tuple[np.ndarray, np.ndarray, float]) -> None:
        """Handle completion of processing."""
        alpha, result_image, timing = result
        
        self.current_alpha = alpha
        self.current_result = result_image
        
        # Update preview
        self._update_preview("-OUTPUT-PREVIEW-", result_image)
        
        # Update UI
        self.window["-PROGRESS-"].update(visible=False)
        self.window["-STATUS-"].update("")
        self.window["-TIMING-INFO-"].update(f"Processed in {timing*1000:.0f}ms")
        self.window["-PROCESS-"].update(disabled=False)
        self.window["-SAVE-"].update(disabled=False)
        
        self.is_processing = False
    
    def _handle_save(self) -> None:
        """Handle saving the result."""
        if self.current_result is None:
            return
        
        # Determine default filename
        if self.current_path:
            default_name = self.current_path.stem + "_matte.png"
            initial_folder = str(self.current_path.parent)
        else:
            default_name = "result.png"
            initial_folder = self.config.get("last_output_folder", "")
        
        # Get save path
        save_path = sg.popup_get_file(
            "Save Result",
            save_as=True,
            default_path=default_name,
            file_types=[
                ("PNG (Transparent)", "*.png"),
                ("JPEG", "*.jpg"),
                ("WebP", "*.webp"),
            ],
            initial_folder=initial_folder,
        )
        
        if not save_path:
            return
        
        try:
            path = Path(save_path)
            
            # Determine format from extension
            if not path.suffix:
                path = path.with_suffix(".png")
            
            # Save image
            save_image(
                self.current_result,
                path,
                quality=self.config.get("jpeg_quality", 95),
            )
            
            # Update status
            self.window["-STATUS-"].update(f"Saved to {path.name}")
            
            # Save folder preference
            self.config.set("last_output_folder", str(path.parent))
            
        except Exception as e:
            sg.popup_error(f"Failed to save:\n\n{e}", title="Error")
    
    def _handle_background_change(self, values: dict) -> None:
        """Handle background option change."""
        bg_option = values["-BACKGROUND-"]
        
        # Enable/disable color picker
        is_custom = bg_option == "Custom Color..."
        self.window["-COLOR-PICKER-"].update(disabled=not is_custom)
        
        # If result exists, recomposite with new background
        if self.current_alpha is not None and self.current_image is not None:
            custom_color = values["-CUSTOM-COLOR-"]
            background = self._get_background_color(bg_option, custom_color)
            self.current_result = apply_alpha_to_image(
                self.current_image, self.current_alpha, background
            )
            self._update_preview("-OUTPUT-PREVIEW-", self.current_result)
    
    def _show_batch_window(self) -> None:
        """Show the batch processing window."""
        layout = create_batch_layout()
        batch_window = sg.Window(
            "Batch Processing",
            layout,
            modal=True,
            finalize=True,
        )
        
        while True:
            event, values = batch_window.read()
            
            if event in (sg.WIN_CLOSED, "-BATCH-CLOSE-", "-BATCH-CANCEL-"):
                break
            
            # Handle batch events here
            # TODO: Implement batch processing logic
        
        batch_window.close()
    
    def _show_settings_window(self) -> None:
        """Show the settings dialog."""
        layout = create_settings_layout(self.config.get_all())
        settings_window = sg.Window(
            "Settings",
            layout,
            modal=True,
            finalize=True,
        )
        
        # Show device info
        if self.engine:
            provider = self.engine.get_active_provider()
            settings_window["-SETTINGS-DEVICE-INFO-"].update(
                f"Currently using: {provider}"
            )
        
        while True:
            event, values = settings_window.read()
            
            if event in (sg.WIN_CLOSED, "-SETTINGS-CANCEL-"):
                break
            
            elif event == "-SETTINGS-SAVE-":
                # Save settings
                self.config.update({
                    "quality": values["-SETTINGS-QUALITY-"].lower(),
                    "background": values["-SETTINGS-BG-"].lower(),
                    "output_format": values["-SETTINGS-FORMAT-"].lower(),
                    "jpeg_quality": int(values["-SETTINGS-JPEG-QUALITY-"]),
                    "auto_save_settings": values["-SETTINGS-AUTOSAVE-"],
                })
                sg.popup_quick_message("Settings saved!", auto_close_duration=2)
                break
            
            elif event == "-SETTINGS-RESET-":
                self.config.reset()
                sg.popup_quick_message("Settings reset to defaults.", auto_close_duration=2)
                break
        
        settings_window.close()
    
    def run(self) -> None:
        """Run the main application event loop."""
        self.window = self.create_main_window()
        
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WIN_CLOSED:
                break
            
            # Handle events
            elif event == "-SELECT-":
                self._handle_file_select()
            
            elif event == "-PROCESS-":
                self._handle_process()
            
            elif event == "-SAVE-":
                self._handle_save()
            
            elif event == "-BACKGROUND-":
                self._handle_background_change(values)
            
            elif event == "-CUSTOM-COLOR-":
                # Custom color selected
                if self.current_alpha is not None:
                    self._handle_background_change(values)
            
            elif event == "-BATCH-":
                self._show_batch_window()
            
            elif event == "-SETTINGS-":
                self._show_settings_window()
            
            elif event == "-ENGINE-PROGRESS-":
                progress, status = values[event]
                self.window["-PROGRESS-"].update(current_count=int(progress * 100))
                self.window["-STATUS-"].update(status)
            
            elif event == "-ENGINE-READY-":
                success, error_msg = values[event]
                self.window["-PROGRESS-"].update(visible=False)
                if success:
                    self.window["-STATUS-"].update("Engine ready")
                    self._run_processing()
                else:
                    self.window["-STATUS-"].update("Engine initialization failed")
                    self.window["-PROCESS-"].update(disabled=False)
                    if error_msg:
                        sg.popup_error(error_msg, title="Initialization Error")
            
            elif event == "-PROCESS-DONE-":
                self._handle_process_complete(values[event])
            
            elif event == "-PROCESS-ERROR-":
                error_msg = values[event]
                self.window["-PROGRESS-"].update(visible=False)
                self.window["-STATUS-"].update("Processing failed")
                self.window["-PROCESS-"].update(disabled=False)
                self.is_processing = False
                sg.popup_error(f"Processing failed:\n\n{error_msg}", title="Error")
            
            # Keyboard shortcuts
            elif event == "-KEY-CTRL-O-":
                self._handle_file_select()
            
            elif event == "-KEY-CTRL-S-":
                if self.current_result is not None:
                    self._handle_save()
            
            elif event in ("-KEY-CTRL-P-", "-KEY-F5-"):
                if self.current_image is not None and not self.is_processing:
                    self._handle_process()
            
            elif event == "-KEY-CTRL-B-":
                self._show_batch_window()
            
            # Window close with confirmation
            elif event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
                break
        
        # Save window position
        if self.window:
            self.config.set("window_position", self.window.current_location())
        
        self.window.close()


def main():
    """Main entry point."""
    app = MattingApp()
    app.run()


if __name__ == "__main__":
    main()
