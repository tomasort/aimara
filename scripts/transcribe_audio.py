#!/usr/bin/env python3
"""Transcribe audio using OpenAI Whisper with GPU/CPU auto-detection."""

import whisper
import torch
from pathlib import Path
import json
import sys
from datetime import datetime

def get_device():
    """Detect available device for processing."""
    print("\n🔍 Detecting available device...\n")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Metal Performance Shaders (MPS) detected - Using Apple GPU")
        print("   (M1/M2/M3 Mac detected)")
        return 'mps'
    
    # Check for CUDA
    elif torch.cuda.is_available():
        print("✅ CUDA detected - Using NVIDIA GPU")
        return 'cuda'
    
    # Fallback to CPU
    else:
        print("⚠️  No GPU detected - Using CPU")
        print("   (This will be slower but will work)")
        return 'cpu'

def transcribe_audio(audio_file, model_name="medium", device=None, language="es"):
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_file: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda, mps) - auto-detected if None
        language: Language code (es for Spanish, en for English, etc.)
        
    Returns:
        Transcription result dictionary
    """
    
    audio_path = Path(audio_file)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    print(f"\n🎙️  Loading Whisper model: {model_name}")
    print(f"   Device: {device.upper()}")
    print(f"   Language: {language}")
    
    try:
        # Load model with specified device
        model = whisper.load_model(model_name, device=device)
        
        print(f"\n📝 Transcribing audio file...")
        print(f"   File: {audio_path.name}")
        print(f"   Size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"\n   This may take several minutes, please wait...\n")
        
        # Transcribe with language specification for better accuracy
        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=True,  # Show progress
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

def save_transcript(result, output_dir="transcripts"):
    """Save transcription results in multiple formats."""
    
    if result is None:
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Use timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"transcript_{timestamp}"
    
    # Save as plain text
    text_file = output_path / f"{base_name}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    print(f"\n✅ Plain text transcript saved:")
    print(f"   {text_file.resolve()}")
    
    # Save as JSON with segments and timestamps
    json_file = output_path / f"{base_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'text': result['text'],
            'language': result.get('language'),
            'segments': result.get('segments', []),
            'duration': result.get('duration'),
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Detailed transcript with timestamps saved:")
    print(f"   {json_file.resolve()}")
    
    # Save as SRT (subtitle format) - useful for video sync
    srt_file = output_path / f"{base_name}.srt"
    segments = result.get('segments', [])
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            
            # Convert seconds to SRT format (HH:MM:SS,mmm)
            start_srt = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
            end_srt = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
            
            f.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")
    
    print(f"✅ Subtitle file (SRT) saved:")
    print(f"   {srt_file.resolve()}")
    
    return {
        'text_file': text_file,
        'json_file': json_file,
        'srt_file': srt_file,
    }

if __name__ == "__main__":
    # Default audio file
    audio_file = "audio/EN VIVO ｜ Con El Mazo Dando ｜ Diosdado Cabello ｜ Programa 548.mp3"
    
    # Allow custom audio file as argument
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    # Allow model selection as second argument
    model = sys.argv[2] if len(sys.argv) > 2 else "medium"
    
    # Allow device override as third argument
    device = sys.argv[3].lower() if len(sys.argv) > 3 else None
    if device not in ['cpu', 'cuda', 'mps', None]:
        print(f"❌ Invalid device: {device}. Use 'cpu', 'cuda', 'mps', or leave empty for auto-detect")
        sys.exit(1)
    
    print("=" * 80)
    print("🎵 Whisper Audio Transcription Tool")
    print("=" * 80)
    
    # Transcribe
    result = transcribe_audio(audio_file, model_name=model, device=device, language="es")
    
    if result:
        # Save results
        files = save_transcript(result)
        
        print("\n" + "=" * 80)
        print("📊 Transcription Statistics:")
        print(f"   Language: {result.get('language', 'unknown')}")
        print(f"   Duration: {result.get('duration', 'unknown')} seconds")
        print(f"   Total segments: {len(result.get('segments', []))}")
        print(f"   Text length: {len(result['text'])} characters")
        print("=" * 80)
        
        sys.exit(0)
    else:
        sys.exit(1)
