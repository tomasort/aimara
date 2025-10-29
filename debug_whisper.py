#!/usr/bin/env python3
"""
Script to pre-download Whisper models and test transcription worker.
This helps debug the hanging issue by isolating the model download step.
"""

import whisper
import torch
import sys
import os
from pathlib import Path

def test_device_detection():
    """Test device detection and availability."""
    print("🔍 Testing device detection...")
    
    # Check CUDA
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
    # Check MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("✅ MPS (Apple Silicon GPU) available")
    else:
        device = "cpu"
        print("⚠️  Using CPU (no GPU acceleration)")
    
    print(f"🎯 Selected device: {device}")
    return device

def download_whisper_models():
    """Pre-download Whisper models to avoid runtime issues."""
    models = ["tiny", "base", "small", "medium"]
    
    print("\n📥 Pre-downloading Whisper models...")
    print("This may take a few minutes but will prevent runtime hangs...")
    
    for model_name in models:
        try:
            print(f"\n⬇️  Downloading {model_name} model...")
            model = whisper.load_model(model_name)
            print(f"✅ {model_name} model downloaded successfully")
            
            # Test the model briefly
            print(f"🧪 Testing {model_name} model...")
            # Create a short silent audio for testing
            import numpy as np
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            result = model.transcribe(test_audio, language="es")
            print(f"✅ {model_name} model test successful")
            
            del model  # Free memory
            
        except Exception as e:
            print(f"❌ Failed to download/test {model_name} model: {e}")
            if "name resolution" in str(e).lower():
                print("🌐 Network/DNS issue detected!")
                print("Solutions:")
                print("   1. Check internet connection")
                print("   2. Try different DNS (8.8.8.8)")
                print("   3. Use VPN if network blocks downloads")
                return False
    
    print("\n✅ All models downloaded and tested successfully!")
    return True

def test_transcription_worker():
    """Test the transcription worker in isolation."""
    print("\n🧪 Testing transcription worker...")
    
    device = test_device_detection()
    
    try:
        print(f"\n📤 Loading tiny model on {device}...")
        model = whisper.load_model("tiny", device=device)
        print("✅ Model loaded successfully")
        
        # Create test audio
        import numpy as np
        test_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of silence
        
        print("🎤 Transcribing test audio...")
        result = model.transcribe(test_audio, language="es")
        print(f"✅ Transcription successful: '{result['text'].strip()}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription worker test failed: {e}")
        
        if "name resolution" in str(e).lower():
            print("\n🌐 DNS/Network issue:")
            print("   - Server can't download Whisper models")
            print("   - Check internet connection")
            print("   - Try: export DNS=8.8.8.8")
            
        elif "out of memory" in str(e).lower():
            print("\n💾 GPU Memory issue:")
            print("   - Try smaller model (tiny instead of medium)")
            print("   - Try CPU: --device cpu")
            
        elif "cuda" in str(e).lower():
            print("\n🎮 CUDA issue:")
            print("   - Check CUDA installation")
            print("   - Try CPU: --device cpu")
            
        return False

def check_network_connectivity():
    """Check if we can reach model download URLs."""
    print("\n🌐 Testing network connectivity...")
    
    import urllib.request
    import socket
    
    urls_to_test = [
        "https://openaipublic.azureedge.net",  # Whisper models
        "https://github.com",                  # General connectivity
        "https://google.com",                  # DNS test
    ]
    
    for url in urls_to_test:
        try:
            print(f"   Testing {url}...")
            response = urllib.request.urlopen(url, timeout=10)
            print(f"   ✅ {url} - OK ({response.code})")
        except Exception as e:
            print(f"   ❌ {url} - FAILED: {e}")
            if "name resolution" in str(e).lower():
                print("      🔧 DNS resolution failed - check DNS settings")
            return False
    
    print("✅ Network connectivity looks good")
    return True

def main():
    print("=" * 80)
    print("🔧 Whisper Model Download & Transcription Test")
    print("=" * 80)
    
    # Test network first
    if not check_network_connectivity():
        print("\n❌ Network issues detected. Fix network connectivity first.")
        sys.exit(1)
    
    # Test device detection
    device = test_device_detection()
    
    # Pre-download models
    if not download_whisper_models():
        print("\n❌ Model download failed. Check error messages above.")
        sys.exit(1)
    
    # Test transcription worker
    if not test_transcription_worker():
        print("\n❌ Transcription worker test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("🚀 Your setup should work with the main pipeline now.")
    print("=" * 80)

if __name__ == "__main__":
    main()