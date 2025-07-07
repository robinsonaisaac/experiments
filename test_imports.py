#!/usr/bin/env python3
"""
Simple test script to check if MFS modules can be imported
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test importing all MFS modules"""
    
    print("Testing MFS module imports...")
    
    try:
        # Test basic imports first
        print("✓ Basic Python imports work")
        
        # Test if we can import the modules structure
        import models
        print("✓ models package imported")
        
        import training
        print("✓ training package imported")
        
        print("\n🎉 All imports successful!")
        print("The MFS implementation structure is working correctly.")
        print("\nTo run full experiments:")
        print("1. Install PyTorch: pip install torch")
        print("2. Install other dependencies: pip install -r requirements.txt")
        print("3. Run: python experiments/train_tiny_local.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nPlease ensure all files are in place:")
        print("- models/__init__.py")
        print("- models/mfs_layer.py")
        print("- models/safety_features.py")
        print("- models/aggregators.py")
        print("- models/full_model.py")
        print("- training/__init__.py")
        print("- training/trainer.py")
        print("- training/losses.py")
        print("- training/datasets.py")
        print("- training/distributed.py")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 