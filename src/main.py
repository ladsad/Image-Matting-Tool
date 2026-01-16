"""Main entry point for the Image Matting Tool."""

import sys
from pathlib import Path

# Add src to path if running directly
if __name__ == "__main__":
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from gui.app import main

if __name__ == "__main__":
    main()
