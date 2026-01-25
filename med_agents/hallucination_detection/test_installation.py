"""
Test Installation Script

Quick test to verify the package is installed correctly.
"""

import sys

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing package imports...")
    
    try:
        # Main package
        import hallucination_detection
        print("✓ hallucination_detection imported successfully")
        print(f"  Version: {hallucination_detection.__version__}")
        
        # Models
        from hallucination_detection.models import BatchEnsembleModel
        print("✓ BatchEnsembleModel imported")
        
        from hallucination_detection.models import FastWeights, BatchEnsembleLinear
        print("✓ FastWeights imported")
        
        from hallucination_detection.models import LoRAAdapter
        print("✓ LoRAAdapter imported")
        
        # Uncertainty
        from hallucination_detection.uncertainty import UncertaintyEstimator
        print("✓ UncertaintyEstimator imported")
        
        from hallucination_detection.uncertainty import UncertaintyMetrics
        print("✓ UncertaintyMetrics imported")
        
        # Detection
        from hallucination_detection.detection import HallucinationDetector
        print("✓ HallucinationDetector imported")
        
        from hallucination_detection.detection import FeatureExtractor
        print("✓ FeatureExtractor imported")
        
        # Training
        from hallucination_detection.training import BatchEnsembleTrainer
        print("✓ BatchEnsembleTrainer imported")
        
        from hallucination_detection.training import SQuADDataset, MMLUDataset
        print("✓ Datasets imported")
        
        # Utils
        from hallucination_detection.utils import load_squad_dataset
        print("✓ load_squad_dataset imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\nTesting dependencies...")
    
    dependencies = [
        "torch",
        "transformers",
        "peft",
        "numpy",
        "sklearn",
        "datasets",
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"❌ {dep} not found")
            all_ok = False
    
    if all_ok:
        print("\n✅ All dependencies installed!")
    else:
        print("\n⚠️ Some dependencies are missing. Run: pip install -r requirements.txt")
    
    return all_ok


def test_package_structure():
    """Test that all expected modules exist."""
    print("\nTesting package structure...")
    
    import os
    import hallucination_detection
    
    package_dir = os.path.dirname(hallucination_detection.__file__)
    
    expected_dirs = [
        "models",
        "uncertainty",
        "detection",
        "training",
        "utils",
    ]
    
    all_ok = True
    for dirname in expected_dirs:
        dirpath = os.path.join(package_dir, dirname)
        if os.path.isdir(dirpath):
            print(f"✓ {dirname}/")
        else:
            print(f"❌ {dirname}/ not found")
            all_ok = False
    
    if all_ok:
        print("\n✅ Package structure is correct!")
    
    return all_ok


def main():
    """Run all tests."""
    print("="*60)
    print("Hallucination Detection Package - Installation Test")
    print("="*60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    print()
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    print()
    
    # Test structure
    results.append(("Structure", test_package_structure()))
    print()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20} {status}")
    
    print()
    
    if all(r[1] for r in results):
        print("🎉 All tests passed! Package is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run: python examples/quickstart.py")
        print("  2. Read: USAGE.md for detailed instructions")
        print("  3. Check: examples/ for more examples")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
