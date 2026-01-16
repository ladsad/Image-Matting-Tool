"""GUI layout definitions for the Image Matting Tool."""

import FreeSimpleGUI as sg

# Color scheme
COLORS = {
    "bg_primary": "#1a1a2e",
    "bg_secondary": "#16213e",
    "accent": "#0f3460",
    "highlight": "#e94560",
    "text": "#eaeaea",
    "text_muted": "#a0a0a0",
    "success": "#4ecca3",
    "warning": "#ffc107",
    "error": "#dc3545",
}

# Font definitions
FONTS = {
    "title": ("Segoe UI", 18, "bold"),
    "heading": ("Segoe UI", 14, "bold"),
    "body": ("Segoe UI", 11),
    "small": ("Segoe UI", 9),
    "button": ("Segoe UI", 11, "bold"),
}


def get_theme():
    """Configure and return PySimpleGUI theme settings."""
    sg.theme("DarkBlue3")
    sg.set_options(
        font=FONTS["body"],
        button_element_size=(12, 1),
        auto_size_buttons=False,
    )


def create_main_layout() -> list:
    """
    Create the main application window layout.
    
    Returns:
        Layout definition for PySimpleGUI.
    """
    # Header section
    header = [
        sg.Column([
            [sg.Text("Image Matting Tool", font=FONTS["title"], pad=(0, 10))],
            [sg.Text("Professional background removal powered by AI", 
                    font=FONTS["small"], text_color=COLORS["text_muted"])],
        ], expand_x=True),
    ]
    
    # Preview area
    preview_frame = sg.Frame(
        "",
        [
            [
                # Input preview with drag-and-drop support
                sg.Column([
                    [sg.Text("Original", font=FONTS["heading"])],
                    [sg.Image(key="-INPUT-PREVIEW-", size=(400, 400), 
                             background_color="#2a2a4a", pad=(5, 5),
                             enable_events=True)],
                    [sg.Text("Drag & drop image here or click 'Select Image'", 
                            font=FONTS["small"], text_color=COLORS["text_muted"],
                            key="-DROP-HINT-")],
                ], element_justification="center"),
                
                # Arrow indicator
                sg.Column([
                    [sg.Text("-->", font=FONTS["title"], pad=(20, 0))],
                ], vertical_alignment="center"),
                
                # Output preview
                sg.Column([
                    [sg.Text("Result", font=FONTS["heading"])],
                    [sg.Image(key="-OUTPUT-PREVIEW-", size=(400, 400),
                             background_color="#2a2a4a", pad=(5, 5))],
                    [sg.Text("", font=FONTS["small"], key="-OUTPUT-HINT-")],
                ], element_justification="center"),
            ],
        ],
        relief=sg.RELIEF_FLAT,
        border_width=0,
        pad=(0, 10),
    )
    
    # Control buttons
    main_buttons = [
        sg.Button("Select Image", key="-SELECT-", size=(14, 1), 
                  font=FONTS["button"], button_color=(COLORS["text"], COLORS["accent"])),
        sg.Button("Remove Background", key="-PROCESS-", size=(18, 1),
                  font=FONTS["button"], button_color=(COLORS["text"], COLORS["highlight"]),
                  disabled=True),
        sg.Button("Save Result", key="-SAVE-", size=(14, 1),
                  font=FONTS["button"], button_color=(COLORS["text"], COLORS["success"]),
                  disabled=True),
    ]
    
    # Settings section
    settings_col1 = sg.Column([
        [sg.Text("Model:", size=(10, 1))],
        [sg.Combo(
            ["MODNet (Fast)", "MODNet Photographic"],
            default_value="MODNet (Fast)",
            key="-MODEL-",
            size=(25, 1),
            readonly=True,
            enable_events=True,
        )],
    ])
    
    settings_col2 = sg.Column([
        [sg.Text("Quality:", size=(10, 1))],
        [sg.Combo(
            ["Standard (Fast)", "High (Balanced)", "Ultra (Best)"],
            default_value="High (Balanced)",
            key="-QUALITY-",
            size=(18, 1),
            readonly=True,
        )],
    ])
    
    settings_col3 = sg.Column([
        [sg.Text("Background:", size=(12, 1))],
        [sg.Combo(
            ["Transparent", "White", "Black", "Custom Color..."],
            default_value="Transparent",
            key="-BACKGROUND-",
            size=(18, 1),
            readonly=True,
            enable_events=True,
        )],
    ])
    
    settings_col4 = sg.Column([
        [sg.Text("", size=(8, 1))],  # Spacer
        [sg.ColorChooserButton(
            "Pick Color",
            key="-COLOR-PICKER-",
            target="-CUSTOM-COLOR-",
            size=(10, 1),
            disabled=True,
        ),
        sg.Input(key="-CUSTOM-COLOR-", visible=False, enable_events=True)],
    ])
    
    settings_frame = sg.Frame(
        "Settings",
        [[settings_col1, settings_col2, settings_col3, settings_col4]],
        relief=sg.RELIEF_GROOVE,
        pad=(0, 10),
    )
    
    # Progress section
    progress_section = [
        sg.Column([
            [sg.ProgressBar(100, key="-PROGRESS-", size=(50, 20), 
                           visible=False, bar_color=(COLORS["success"], COLORS["bg_secondary"]))],
            [sg.Text("", key="-STATUS-", size=(60, 1), font=FONTS["small"],
                    text_color=COLORS["text_muted"], justification="center")],
        ], element_justification="center", expand_x=True),
    ]
    
    # Info section
    info_section = [
        sg.Column([
            [sg.Text("", key="-FILE-INFO-", font=FONTS["small"], 
                    text_color=COLORS["text_muted"])],
            [sg.Text("", key="-TIMING-INFO-", font=FONTS["small"],
                    text_color=COLORS["success"])],
        ], expand_x=True),
    ]
    
    # Bottom toolbar
    toolbar = [
        sg.Button("Batch Mode", key="-BATCH-", size=(12, 1)),
        sg.Button("Settings", key="-SETTINGS-", size=(10, 1)),
        sg.Push(),
        sg.Text("v0.1.0", font=FONTS["small"], text_color=COLORS["text_muted"]),
    ]
    
    # Combine all sections
    layout = [
        header,
        [preview_frame],
        [sg.HorizontalSeparator(pad=(0, 10))],
        [sg.Column([main_buttons], element_justification="center", expand_x=True)],
        [settings_frame],
        progress_section,
        info_section,
        [sg.HorizontalSeparator(pad=(10, 5))],
        toolbar,
    ]
    
    return layout


def create_batch_layout() -> list:
    """
    Create the batch processing window layout.
    
    Returns:
        Layout definition for batch processing dialog.
    """
    layout = [
        [sg.Text("Batch Processing", font=FONTS["title"])],
        [sg.Text("Process multiple images at once", font=FONTS["small"], 
                text_color=COLORS["text_muted"])],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        # Folder selection
        [sg.Text("Input Folder:", size=(12, 1))],
        [sg.Input(key="-BATCH-INPUT-", size=(50, 1), readonly=True, enable_events=True),
         sg.FolderBrowse("Browse", key="-BATCH-INPUT-BROWSE-")],
        
        [sg.Text("Output Folder:", size=(12, 1))],
        [sg.Input(key="-BATCH-OUTPUT-", size=(50, 1), readonly=True),
         sg.FolderBrowse("Browse", key="-BATCH-OUTPUT-BROWSE-")],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        # Settings
        [sg.Text("Quality:", size=(12, 1)),
         sg.Combo(["Standard", "High", "Ultra"], default_value="High",
                  key="-BATCH-QUALITY-", size=(15, 1), readonly=True)],
        
        [sg.Text("Background:", size=(12, 1)),
         sg.Combo(["Transparent", "White", "Black"], default_value="Transparent",
                  key="-BATCH-BG-", size=(15, 1), readonly=True)],
        
        [sg.Text("Output Format:", size=(12, 1)),
         sg.Combo(["PNG", "JPEG", "WebP"], default_value="PNG",
                  key="-BATCH-FORMAT-", size=(15, 1), readonly=True)],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        # File list
        [sg.Text("Files to process:", font=FONTS["heading"])],
        [sg.Listbox([], key="-BATCH-FILES-", size=(60, 10), 
                   select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
        [sg.Text("0 files selected", key="-BATCH-COUNT-", font=FONTS["small"])],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        # Progress
        [sg.ProgressBar(100, key="-BATCH-PROGRESS-", size=(50, 20), visible=False)],
        [sg.Text("", key="-BATCH-STATUS-", size=(60, 1), font=FONTS["small"])],
        
        # Buttons
        [sg.Button("Start Processing", key="-BATCH-START-", size=(16, 1), 
                  button_color=(COLORS["text"], COLORS["highlight"])),
         sg.Button("Cancel", key="-BATCH-CANCEL-", size=(10, 1)),
         sg.Push(),
         sg.Button("Close", key="-BATCH-CLOSE-", size=(10, 1))],
    ]
    
    return layout


def create_settings_layout(current_settings: dict) -> list:
    """
    Create the settings dialog layout.
    
    Args:
        current_settings: Current application settings.
    
    Returns:
        Layout definition for settings dialog.
    """
    layout = [
        [sg.Text("Settings", font=FONTS["title"])],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        # Hardware settings
        [sg.Frame("Hardware", [
            [sg.Text("Inference Device:", size=(15, 1)),
             sg.Combo(["Auto (GPU if available)", "GPU Only", "CPU Only"],
                     default_value="Auto (GPU if available)",
                     key="-SETTINGS-DEVICE-", size=(25, 1), readonly=True)],
            [sg.Text("", key="-SETTINGS-DEVICE-INFO-", font=FONTS["small"],
                    text_color=COLORS["text_muted"])],
        ], pad=(0, 10))],
        
        # Default settings
        [sg.Frame("Defaults", [
            [sg.Text("Default Quality:", size=(15, 1)),
             sg.Combo(["Standard", "High", "Ultra"],
                     default_value=current_settings.get("quality", "high").title(),
                     key="-SETTINGS-QUALITY-", size=(15, 1), readonly=True)],
            [sg.Text("Default Background:", size=(15, 1)),
             sg.Combo(["Transparent", "White", "Black"],
                     default_value=current_settings.get("background", "transparent").title(),
                     key="-SETTINGS-BG-", size=(15, 1), readonly=True)],
            [sg.Text("Output Format:", size=(15, 1)),
             sg.Combo(["PNG", "JPEG", "WebP"],
                     default_value=current_settings.get("output_format", "png").upper(),
                     key="-SETTINGS-FORMAT-", size=(15, 1), readonly=True)],
            [sg.Text("JPEG Quality:", size=(15, 1)),
             sg.Slider((50, 100), default_value=current_settings.get("jpeg_quality", 95),
                      key="-SETTINGS-JPEG-QUALITY-", orientation="h", size=(20, 15))],
        ], pad=(0, 10))],
        
        # Behavior settings
        [sg.Frame("Behavior", [
            [sg.Checkbox("Auto-save settings on exit", 
                        default=current_settings.get("auto_save_settings", True),
                        key="-SETTINGS-AUTOSAVE-")],
            [sg.Checkbox("Remember last used folders",
                        default=True,
                        key="-SETTINGS-REMEMBER-FOLDERS-")],
        ], pad=(0, 10))],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        # Buttons
        [sg.Button("Save", key="-SETTINGS-SAVE-", size=(10, 1),
                  button_color=(COLORS["text"], COLORS["success"])),
         sg.Button("Reset to Defaults", key="-SETTINGS-RESET-", size=(16, 1)),
         sg.Push(),
         sg.Button("Cancel", key="-SETTINGS-CANCEL-", size=(10, 1))],
    ]
    
    return layout


def create_about_layout() -> list:
    """Create the about dialog layout."""
    layout = [
        [sg.Text("Image Matting Tool", font=FONTS["title"])],
        [sg.Text("Version 0.1.0", font=FONTS["body"])],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        [sg.Text("Professional background removal using AI-powered image matting.",
                font=FONTS["body"])],
        
        [sg.Text("")],
        
        [sg.Text("Powered by:", font=FONTS["heading"])],
        [sg.Text("  - MODNet (Trimap-Free Portrait Matting)", font=FONTS["small"])],
        [sg.Text("  - ONNX Runtime for inference", font=FONTS["small"])],
        [sg.Text("  - PySimpleGUI for interface", font=FONTS["small"])],
        
        [sg.Text("")],
        
        [sg.Text("Licensed under Apache 2.0", font=FONTS["small"], 
                text_color=COLORS["text_muted"])],
        
        [sg.HorizontalSeparator(pad=(0, 10))],
        
        [sg.Button("OK", key="-ABOUT-OK-", size=(10, 1))],
    ]
    
    return layout
