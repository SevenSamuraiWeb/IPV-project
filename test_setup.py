import os
import sys
import numpy as np


def test_imports():
    print("Testing imports...")
    try:

        import torch
        import cv2
        import numpy
        import PIL
        from torchvision import models
        print("[OK] Standard libraries imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    return True

def test_model_instantiation():
    print("\nTesting Model Instantiation...")
    try:
        # We need to temporarily add the project dir to path to import model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.join(current_dir, "../../Downloads/IPV-project") 
        # Adjust path if script is run from a different location, assuming standard agent structure involves knowing absolute paths
        # Let's use the known absolute path from the prompt context
        project_path = r"c:\Users\nihaa\Downloads\IPV-project"
        sys.path.append(project_path)
        
        import torch
        from model import SaliencyModel
        model = SaliencyModel()
        print("[OK] SaliencyModel instantiated successfully.")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"[OK] Forward pass successful. Output shape: {output.shape}")
        
    except Exception as e:
        print(f"[FAIL] Model test failed: {e}")
        import traceback
        traceback.print_exc()

def test_utils_logic():
    print("\nTesting Utils Logic...")
    try:
        sys.path.append(r"c:\Users\nihaa\Downloads\IPV-project")
        # We need to bypass the 'import streamlit' inside utils.py
        # Since we mocked sys.modules['streamlit'], it should be fine.
        from utils import refine_mask
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255 # Square in middle
        refined = refine_mask(mask)
        print("[OK] refine_mask execution successful.")
        
    except Exception as e:
        print(f"[FAIL] Utils test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if test_imports():
        test_model_instantiation()
        test_utils_logic()
