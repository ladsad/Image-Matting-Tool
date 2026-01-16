import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))
from src.utils.paths import get_cache_dir

def probe_rvm_scales():
    model_path = get_cache_dir() / 'models' / 'rvm_resnet50_fp32.onnx'
    print(f"Probing {model_path}...")
    
    if not model_path.exists():
        print("Model not found")
        return

    sess = ort.InferenceSession(str(model_path))
    
    # Setup consistent input
    h, w = 512, 512 # Standard size divisible by 32
    src = np.zeros((1, 3, h, w), dtype=np.float32)
    downsample = np.array([0.25], dtype=np.float32)
    
    # Defined channels from inspection
    channels = {"r1i": 16, "r2i": 32, "r3i": 64, "r4i": 128}
    
    # Test cases: tuples of scales for (r1, r2, r3, r4)
    # Scale S means dimension is Dim // S
    test_scales = [
        (1, 2, 4, 8),    # My current config
        (2, 4, 8, 16),   # Likely alternative (stride 2 start)
        (4, 8, 16, 32),  # Deeper start
        (1, 1, 1, 1),    # All full res?
    ]
    
    with open("probe_results.txt", "w") as log:
        for scales in test_scales:
            print(f"Testing scales: {scales}")
            log.write(f"Testing scales: {scales}\n")
            try:
                inputs = {
                    "src": src,
                    "downsample_ratio": downsample
                }
                
                # Construct recurrent inputs
                scale_map = {
                    "r1i": scales[0], 
                    "r2i": scales[1], 
                    "r3i": scales[2], 
                    "r4i": scales[3]
                }
                
                for name, ch in channels.items():
                    scale = scale_map[name]
                    inputs[name] = np.zeros((1, ch, h // scale, w // scale), dtype=np.float32)
                    
                sess.run(None, inputs)
                print("SUCCESS!")
                log.write("SUCCESS! These scales work.\n")
                return scales
                
            except Exception as e:
                print(f"Failed")
                log.write(f"Failed with: {e}\n")

if __name__ == "__main__":
    probe_rvm_scales()
