# Transcription Script Usage Guide

## Overview
The `transcribe_audio.py` script uses OpenAI's Whisper to convert audio to text with automatic device detection.

## Features
- ✅ **Auto GPU Detection**: Automatically uses MPS (Apple Silicon) or CUDA (NVIDIA) if available
- ✅ **CPU Fallback**: Works on Intel Macs with CPU processing
- ✅ **Spanish Language Optimized**: Pre-configured for Spanish language transcription
- ✅ **Multiple Output Formats**:
  - Plain text (.txt) - just the transcript
  - JSON (.json) - includes timestamps and segments
  - SRT (.srt) - subtitle format for video syncing

## Whisper Models (for Spanish)

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| tiny | ⚡⚡⚡⚡⚡ Very Fast | Low | Quick testing |
| base | ⚡⚡⚡ Fast | Good | **Intel Mac (recommended)** |
| small | ⚡⚡ Medium | Very Good | Better accuracy needed |
| medium | ⚡ Slow | Excellent | **Best accuracy (your choice)** |
| large | 🐌 Very Slow | Best | When accuracy is critical |

## Usage

### Basic usage (default: medium model, Spanish, auto-detect device)
```bash
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py
```

### Specify different audio file
```bash
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py audio/my_file.mp3
```

### Use different model size
```bash
# Use base model instead (faster on Intel Mac)
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py audio/my_file.mp3 base

# Use large model (best accuracy, slower)
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py audio/my_file.mp3 large
```

### Force specific device
```bash
# Force CPU (even if GPU available)
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py audio/my_file.mp3 medium cpu

# Force MPS (if on M1/M2/M3 Mac)
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py audio/my_file.mp3 medium mps

# Force CUDA (if NVIDIA GPU available)
docker-compose run --rm youtube-processor python scripts/transcribe_audio.py audio/my_file.mp3 medium cuda
```

## Performance on Intel Mac (2019)

Estimated times for 4-hour audio file:

- `base` model: 15-30 minutes
- `small` model: 30-60 minutes
- `medium` model: 60-90 minutes (your choice - best Spanish accuracy)

## Output Files

All files are saved to `transcripts/` directory with timestamp:

1. **transcript_YYYYMMDD_HHMMSS.txt**
   - Plain text of entire transcript
   - Use this for searching/grepping keywords

2. **transcript_YYYYMMDD_HHMMSS.json**
   - Structured data with timestamps
   - Segments showing start/end times
   - Use for advanced analysis

3. **transcript_YYYYMMDD_HHMMSS.srt**
   - Subtitle format
   - Use for syncing with video

## Tips for Spanish Transcription

- The script auto-detects language as Spanish (`language="es"`)
- `medium` model is recommended for Spanish due to accent/dialect variations
- For very long videos (8+ hours), consider breaking into chunks
- MPS on Apple Silicon (M1/M2/M3) will be ~2-3x faster than CPU

## Next Steps

After transcription completes:
1. Search the `.txt` transcript for your keywords using regex
2. Use the `.json` file to get exact timestamps for matches
3. Use the `.srt` file to verify context in the video

## Troubleshooting

**"Out of memory" error?**
- Use smaller model (base instead of medium)
- Break audio into smaller chunks

**Transcription is slow?**
- This is normal for CPU processing
- Can't speed up without GPU

**Poor accuracy for Spanish?**
- Ensure model is at least "medium"
- Poor audio quality will affect accuracy
