import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))
from src.utils.paths import get_cache_dir

def install_model():
    source = Path("temp_matteformer/matteformer.onnx")
    if not source.exists():
        print("Source model not found!")
        return

    dest_dir = get_cache_dir() / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "matteformer.onnx"
    
    print(f"Installing {source} to {dest}...")
    shutil.copy2(source, dest)
    print("Installation complete.")

if __name__ == "__main__":
    install_model()
