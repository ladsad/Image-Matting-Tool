"""Main entry point for the Image Matting Tool."""

import sys
from pathlib import Path

# Add project root to path for imports to work correctly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.gui.app import main

if __name__ == "__main__":
    main()
