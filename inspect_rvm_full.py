import onnxruntime as ort
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))
from src.utils.paths import get_cache_dir

def inspect_model_to_file(filename):
    path = get_cache_dir() / 'models' / filename
    output_file = Path(f'inspect_{filename}.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"--- Inspecting {filename} ---\n")
        if not path.exists():
            f.write("File not found\n")
            return

        try:
            sess = ort.InferenceSession(str(path))
            f.write("INPUTS:\n")
            for i in sess.get_inputs():
                f.write(f"  {i.name}: {i.shape}, {i.type}\n")
            
            f.write("\nOUTPUTS:\n")
            for o in sess.get_outputs():
                f.write(f"  {o.name}: {o.shape}, {o.type}\n")
                
        except Exception as e:
            f.write(f"Error: {e}\n")
    
    print(f"Written inspection to {output_file}")

inspect_model_to_file('rvm_resnet50_fp32.onnx')
inspect_model_to_file('rvm_mobilenetv3_fp32.onnx')
