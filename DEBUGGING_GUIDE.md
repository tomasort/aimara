# Quick Fix Guide for Transcription Hanging

## Problem Identified
Your transcription worker is failing with DNS resolution error when trying to download Whisper models:

```
TranscriptionWorker] ERROR    Transcription worker fatal error: <urlopen error [Errno -3] Temporary failure in name resolution>
```

## Solutions (in order of preference):

### 1. Pre-download Models (Recommended)
Run this first to download models when network is stable:

```bash
python debug_whisper.py
```

This will:
- Test network connectivity  
- Download all Whisper models (tiny, base, small, medium)
- Test each model works
- Give specific error messages

### 2. Network/DNS Fixes

**Try different DNS:**
```bash
export DNS=8.8.8.8
# or
export DNS=1.1.1.1
```

**Check if server blocks model downloads:**
```bash
curl -I https://openaipublic.azureedge.net
```

### 3. Use Local Model Files
If network issues persist, manually download models:

```bash
# Download models to cache manually
wget https://openaipublic.azureedge.net/whisper/models/tiny.pt
mv tiny.pt ~/.cache/whisper/
```

### 4. Add Timeout/Retry Logic
Add this to your transcription worker initialization:

```python
import whisper
import time
import sys

def load_model_with_retry(model_name, device, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Loading {model_name} (attempt {attempt + 1}/{max_retries})")
            model = whisper.load_model(model_name, device=device)
            print(f"✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("❌ All attempts failed. Check network connectivity.")
                sys.exit(1)
```

### 5. Pipeline Improvements Needed

Your current pipeline needs these fixes:

1. **Better error handling**: Don't hang when transcription worker dies
2. **Model pre-loading**: Load models before starting worker threads  
3. **Graceful shutdown**: Stop producer when worker fails
4. **Progress monitoring**: Show what's happening during model loading

## Next Steps

1. Run `python debug_whisper.py` first
2. If that passes, your main pipeline should work
3. If debug fails, fix network issues first
4. Consider adding retry logic to your main code

The hanging happens because your producer keeps making chunks but the transcription worker died early, so nothing processes the queue.