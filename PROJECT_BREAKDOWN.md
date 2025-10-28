# YouTube Playlist Video Processing Project

## Project Overview
This project aims to download videos from a YouTube playlist, transcribe them using OpenAI's Whisper, and search for specific keywords within the transcripts.

## Project Breakdown

### 1. Video Download Phase
**Goal**: Extract and download all videos from a YouTube playlist

**Key Components**:
- **Playlist URL parsing**: Extract video URLs from playlist
- **Video downloading**: Use yt-dlp (recommended over youtube-dl as it's more actively maintained)
- **Format selection**: Choose appropriate video/audio quality for processing

**Implementation Focus Areas**:
- **Rate limiting**: YouTube has rate limits; implement delays between downloads
- **Error handling**: Handle network failures, unavailable videos, age-restricted content
- **Storage management**: Videos can be large; consider disk space and cleanup strategies
- **Metadata extraction**: Capture video titles, descriptions, upload dates for reference

**Recommended Tools**:
- `yt-dlp` - Most reliable YouTube downloader
- `requests` - For API calls if needed
- `pathlib` - For file path management

### 2. Audio Extraction Phase
**Goal**: Extract audio from downloaded videos for transcription

**Key Components**:
- **Format conversion**: Convert video to audio (WAV/MP3)
- **Audio optimization**: Prepare audio for Whisper (sample rate, channels)

**Implementation Focus Areas**:
- **Audio quality**: Balance between file size and transcription accuracy
- **Batch processing**: Handle multiple files efficiently
- **Temporary file management**: Clean up intermediate files

**Recommended Tools**:
- `ffmpeg` - Industry standard for audio/video processing
- `pydub` - Python wrapper for audio manipulation

### 3. Transcription Phase
**Goal**: Convert audio to text using OpenAI Whisper

**Key Components**:
- **Whisper model selection**: Choose appropriate model size (tiny, base, small, medium, large)
- **Batch transcription**: Process multiple audio files
- **Output formatting**: Structure transcripts with timestamps

**Implementation Focus Areas**:
- **Model performance vs speed**: Larger models are more accurate but slower
- **Memory management**: Large models require significant RAM
- **GPU utilization**: Use CUDA if available for faster processing
- **Timestamp preservation**: Keep timing information for context
- **Language detection**: Handle multiple languages if needed

**Recommended Approach**:
```python
import whisper

# Load model once
model = whisper.load_model("base")

# Process each audio file
for audio_file in audio_files:
    result = model.transcribe(audio_file)
    # Save transcript with timestamps
```

### 4. Search and Analysis Phase
**Goal**: Search transcripts for specific keywords or patterns

**Key Components**:
- **Regex pattern matching**: Find keyword occurrences
- **Context extraction**: Get surrounding text for relevant matches
- **Result formatting**: Present findings with video source and timestamps

**Implementation Focus Areas**:
- **Search accuracy**: Handle variations in speech-to-text conversion
- **Context window**: Provide sufficient context around matches
- **Case sensitivity**: Consider different capitalizations
- **Fuzzy matching**: Account for transcription errors
- **Result ranking**: Order by relevance or confidence

**Advanced Search Features**:
- **Phonetic matching**: Handle similar-sounding words
- **Stemming/Lemmatization**: Find word variations
- **Sentiment analysis**: Understand context around keywords

## Docker Implementation Strategy

### Recommended Approach: Custom Python Image
**Best choice**: Use official Python image and install dependencies rather than yt-dlp-specific images.

**Reasoning**:
- **Flexibility**: You need multiple tools (yt-dlp, whisper, ffmpeg, python libraries)
- **Control**: Custom image allows precise dependency management
- **GPU support**: Easier to add CUDA support for Whisper acceleration
- **Maintenance**: Single image vs. multiple specialized containers

### Docker Architecture Options

#### Option 1: Single Container (Recommended for Development)
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
```

**Pros**: Simple, all tools in one place, easy development
**Cons**: Larger image size, less modular

#### Option 2: Multi-Container Setup (Production Ready)
```yaml
version: '3.8'
services:
  downloader:
    build: ./docker/downloader
    volumes:
      - ./downloads:/app/downloads
      - ./config:/app/config
    
  transcriber:
    build: ./docker/transcriber
    volumes:
      - ./downloads:/app/downloads
      - ./transcripts:/app/transcripts
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  searcher:
    build: ./docker/searcher
    volumes:
      - ./transcripts:/app/transcripts
      - ./results:/app/results
```

**Pros**: Modular, scalable, resource optimization
**Cons**: More complex, requires orchestration

### Recommended Base Images

#### For Single Container Approach:
1. **`python:3.11-slim`** (Recommended)
   - Minimal size while including build tools
   - Good balance of size vs. functionality
   - Easy to add system dependencies

2. **`python:3.11`** (If you need more tools)
   - Includes more development tools
   - Larger but more convenient for complex builds

#### For GPU Acceleration:
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

### Docker Configuration Files

#### Dockerfile (Single Container with uv for Performance)
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv - fast Python package installer written in Rust
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies with uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p downloads audio transcripts results

# Set permissions
RUN chmod +x scripts/*.py

CMD ["python", "main.py"]
```

**Why `uv` instead of `pip`?**:
- **⚡ Speed**: 10-100x faster package installation
- **🔒 Reliable**: Better dependency resolution
- **💾 Efficient**: Reduced Docker layer size
- **📦 Lock files**: Optional `uv.lock` for reproducible builds

#### docker-compose.yml
```yaml
version: '3.8'

services:
  youtube-processor:
    build: .
    container_name: aimara-processor
    volumes:
      - ./downloads:/app/downloads
      - ./audio:/app/audio
      - ./transcripts:/app/transcripts
      - ./results:/app/results
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
    stdin_open: true
    tty: true
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

#### .dockerignore
```
downloads/
audio/
transcripts/
results/
__pycache__/
*.pyc
.git/
.gitignore
README.md
PROJECT_BREAKDOWN.md
```

## Technical Architecture

### Directory Structure
```
project/
├── downloads/          # Raw video files (mounted volume)
├── audio/             # Extracted audio files (mounted volume)
├── transcripts/       # Text transcripts (mounted volume)
├── results/           # Search results (mounted volume)
├── docker/            # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   ├── download.py    # Video downloading
│   ├── extract.py     # Audio extraction
│   ├── transcribe.py  # Whisper transcription
│   └── search.py      # Keyword searching
├── config/
│   └── settings.json  # Configuration file
├── requirements.txt   # Python dependencies
└── main.py           # Main orchestrator
```

### Configuration Management
Create a configuration file to manage:
- YouTube playlist URLs
- Download preferences (quality, format)
- Whisper model selection
- Search keywords and patterns
- Output preferences

### Error Handling Strategy
1. **Graceful degradation**: Continue processing even if some videos fail
2. **Detailed logging**: Track progress and errors for debugging
3. **Resume capability**: Allow restarting from where process stopped
4. **Validation checks**: Verify downloads and transcriptions

### Performance Optimization
1. **Parallel processing**: Download and process multiple files simultaneously
2. **Caching**: Store intermediate results to avoid reprocessing
3. **Progressive processing**: Start transcription while downloads continue
4. **Resource monitoring**: Track CPU, memory, and disk usage

### Docker Advantages for This Project

#### Dependency Isolation
- **System dependencies**: ffmpeg, CUDA drivers (if using GPU)
- **Python environment**: Avoid conflicts with system Python
- **Version control**: Lock specific versions of yt-dlp, whisper, etc.

#### Resource Management
- **Memory limits**: Prevent Whisper from consuming all system RAM
- **CPU limits**: Control resource usage during processing
- **Storage**: Easy cleanup of intermediate files

#### Scalability Benefits
- **Horizontal scaling**: Run multiple containers for parallel processing
- **Cloud deployment**: Easy deployment to cloud platforms
- **Development consistency**: Same environment across different machines

### Docker Best Practices for This Project

#### Volume Management
```bash
# Create named volumes for persistent data
docker volume create aimara-downloads
docker volume create aimara-results

# Use bind mounts for configuration
-v $(pwd)/config:/app/config:ro
```

#### Memory Optimization
```yaml
# In docker-compose.yml
services:
  transcriber:
    deploy:
      resources:
        limits:
          memory: 8G  # Adjust based on Whisper model
        reservations:
          memory: 4G
```

#### Development vs Production
```dockerfile
# Multi-stage build for production
FROM python:3.11-slim AS base
# ... base setup

FROM base AS development
RUN pip install pytest black flake8
CMD ["python", "-m", "pytest"]

FROM base AS production
CMD ["python", "main.py"]
```

## Core Libraries
```python
# Video downloading
yt-dlp

# Audio processing
ffmpeg-python
pydub

# Transcription
openai-whisper
torch  # For GPU acceleration

# Text processing
regex
nltk  # For advanced text analysis

# Utilities
pathlib
logging
json
concurrent.futures
```

### System Requirements (Docker)
- **Docker 20.10+** and **Docker Compose 2.0+**
- **8GB+ RAM** (for medium/large Whisper models)
- **nvidia-docker2** (optional, for GPU acceleration)
- **Sufficient disk space** (videos can be 100MB-1GB each)
- **Good internet connection** (for downloads)

### Docker Commands for Development
```bash
# Build and start services
docker-compose up --build

# Run specific commands
docker-compose exec youtube-processor python scripts/download.py

# View logs
docker-compose logs -f youtube-processor

# Shell access
docker-compose exec youtube-processor bash

# Cleanup
docker-compose down -v
```

## Potential Challenges and Solutions

### 1. YouTube Restrictions
**Challenge**: Rate limiting, geo-blocking, format changes
**Solution**: 
- Implement exponential backoff
- Use proxy rotation if needed
- Keep yt-dlp updated
- Have fallback strategies

### 2. Large File Handling
**Challenge**: Processing GB of video data
**Solution**:
- Stream processing where possible
- Implement cleanup routines
- Use efficient storage formats
- Consider cloud storage for large datasets

### 3. Transcription Accuracy
**Challenge**: Background noise, accents, technical terms
**Solution**:
- Use appropriate Whisper model size
- Preprocess audio (noise reduction)
- Implement confidence scoring
- Manual review for critical searches

### 4. Search Precision
**Challenge**: Transcription errors affecting search results
**Solution**:
- Use fuzzy matching algorithms
- Implement multiple search strategies
- Provide confidence scores
- Allow manual verification

## Development Phases

### Phase 1: Core Functionality
1. Basic video downloading from playlist
2. Audio extraction
3. Simple transcription
4. Basic keyword search

### Phase 2: Robustness
1. Error handling and recovery
2. Progress tracking and resumption
3. Configuration management
4. Logging and monitoring

### Phase 3: Optimization
1. Parallel processing
2. GPU acceleration
3. Advanced search features
4. Result visualization

### Phase 4: Enhancement
1. Web interface
2. Real-time processing
3. Advanced analytics
4. Export capabilities

## Success Metrics
- **Download success rate**: >95% of playlist videos successfully downloaded
- **Transcription accuracy**: Subjective evaluation of transcript quality
- **Search recall**: Ability to find known keyword occurrences
- **Processing speed**: Time from playlist URL to search results
- **Resource efficiency**: CPU, memory, and disk usage optimization

## Getting Started Checklist
- [ ] Set up Python environment with required dependencies
- [ ] Install system dependencies (ffmpeg)
- [ ] Test yt-dlp with sample playlist
- [ ] Verify Whisper installation and model download
- [ ] Create basic project structure
- [ ] Implement minimal viable version
- [ ] Test with small dataset first
- [ ] Scale up gradually

This breakdown provides a solid foundation for your YouTube playlist processing project. Start with a minimal implementation and gradually add features as you validate each component works correctly.