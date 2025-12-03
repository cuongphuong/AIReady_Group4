"""
Test Llama model loading
"""
import sys
import psutil
from services.llama_service import get_llama_service

def check_system():
    """Check system resources"""
    print("="*80)
    print("SYSTEM CHECK")
    print("="*80)
    
    # Check RAM
    mem = psutil.virtual_memory()
    print(f"💾 Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"💾 Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"💾 Used RAM: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    
    if mem.available < 16 * (1024**3):
        print("⚠️  WARNING: Less than 16GB RAM available. Model loading may fail!")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("⚠️  No GPU available, will use CPU (requires 16GB+ RAM)")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    print("="*80)
    print()

def test_llama_service():
    """Test loading Llama service"""
    print("="*80)
    print("TESTING LLAMA SERVICE")
    print("="*80)
    
    try:
        print("1️⃣ Getting Llama service instance...")
        service = get_llama_service()
        print(f"   ✅ Service instance created")
        print(f"   📍 Model path: {service.model_path}")
        print(f"   💻 Running on: CPU (GGUF format)")
        print()
        
        print("2️⃣ Loading Llama model...")
        print("   ⏳ This may take several minutes...")
        service.load_model()
        print("   ✅ Model loaded successfully!")
        print()
        
        print("3️⃣ Testing classification...")
        result = service.classify_bug(
            "Button không click được",
            ["UI", "Backend", "Database", "Performance"],
            []
        )
        print(f"   ✅ Classification result: {result}")
        print()
        
        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        return True
        
    except MemoryError as e:
        print(f"   ❌ Memory Error: {e}")
        print()
        print("💡 SOLUTIONS:")
        print("   1. Close other applications to free up RAM")
        print("   2. Use a GPU if available (requires CUDA)")
        print("   3. Use a smaller model")
        print("   4. Upgrade system RAM to 16GB+")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    check_system()
    success = test_llama_service()
    sys.exit(0 if success else 1)
