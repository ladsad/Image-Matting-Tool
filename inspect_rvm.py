import onnxruntime as ort
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))
from src.utils.paths import get_cache_dir

def inspect_model(filename):
    path = get_cache_dir() / 'models' / filename
    print(f"\n--- Inspecting {filename} ---")
    if not path.exists():
        print("File not found")
        return

    try:
        sess = ort.InferenceSession(str(path))
        for i in sess.get_inputs():
            if i.name.startswith('r') and i.name.endswith('i'):
                 # shape is [batch, channels, h, w]
                 print(f"{i.name} channels: {i.shape[1]}")
    except Exception as e:
        print(f"Error: {e}")

inspect_model('rvm_resnet50_fp32.onnx')
inspect_model('rvm_mobilenetv3_fp32.onnx')
