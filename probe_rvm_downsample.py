import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))
from src.utils.paths import get_cache_dir

def probe_rvm_downsample():
    model_path = get_cache_dir() / 'models' / 'rvm_resnet50_fp32.onnx'
    print(f"Probing {model_path} with Downsample Logic...")
    
    if not model_path.exists():
        print("Model not found")
        return

    sess = ort.InferenceSession(str(model_path))
    
    # Setup consistent input
    h, w = 512, 512
    src = np.zeros((1, 3, h, w), dtype=np.float32)
    
    # Try different downsample ratios
    ds_ratio = 0.25
    downsample = np.array([ds_ratio], dtype=np.float32)
    
    # Calculate base size for recurrent states (Target Size)
    # The model likely processes at H * ds, W * ds
    base_h = int(h * ds_ratio)
    base_w = int(w * ds_ratio)
    
    print(f"Base Internal Resolution: {base_h}x{base_w}")
    
    # Channels
    channels = {"r1i": 16, "r2i": 32, "r3i": 64, "r4i": 128}
    
    # Test cases: tuples of scales relative to BASE size
    test_scales = [
        (1, 2, 4, 8),    # Standard relative to base?
        (1, 1, 1, 1),    # Same as base?
        (2, 4, 8, 16),
    ]
    
    with open("probe_ds_results.txt", "w") as log:
        for scales in test_scales:
            print(f"Testing scales relative to {base_h}x{base_w}: {scales}")
            log.write(f"Testing relative to {base_h}x{base_w}: {scales}\n")
            try:
                inputs = {
                    "src": src,
                    "downsample_ratio": downsample
                }
                
                scale_map = {
                    "r1i": scales[0], "r2i": scales[1], 
                    "r3i": scales[2], "r4i": scales[3]
                }
                
                for name, ch in channels.items():
                    scale = scale_map[name]
                    # Compute size based on DOWN-SAMPLED base
                    rec_h = base_h // scale
                    rec_w = base_w // scale
                    inputs[name] = np.zeros((1, ch, rec_h, rec_w), dtype=np.float32)
                    print(f"  {name}: {rec_h}x{rec_w}")
                    
                sess.run(None, inputs)
                print("SUCCESS!")
                log.write("SUCCESS!\n")
                return scales
                
            except Exception as e:
                print(f"Failed: {e}")
                log.write(f"Failed: {e}\n")

if __name__ == "__main__":
    probe_rvm_downsample()
