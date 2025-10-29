#!/usr/bin/env python3
"""
Aimara Pipeline - Producer-Consumer architecture for audio processing
======================================================================

Multi-threaded pipeline for:
1. DownloadProducer: Downloads videos and produces audio files
2. TranscriptionWorker: Transcribes audio and produces segments
3. EntityExtractor: Extracts entities from segments and saves to file

Uses thread-safe queues for communication between stages.
"""

import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import subprocess
import sys

import click
import yt_dlp
import whisper
import torch
import spacy
from collections import defaultdict, Counter
import hashlib
import os

# Configure logging with stream handler that flushes immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Add a stream handler with flush
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = FlushingStreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s'))
logger.addHandler(handler)

# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class AudioFile:
    """Represents an audio file to be transcribed."""
    path: str
    video_id: str
    title: str
    duration: Optional[float] = None
    chunk_number: Optional[int] = None  # Which chunk this is (0, 1, 2, ...)
    chunk_length: Optional[int] = None  # Length of each chunk in seconds
    overlap_seconds: Optional[int] = None  # Overlap between chunks
    
    def get_time_offset(self) -> float:
        """Calculate the time offset for this chunk in the original video."""
        if self.chunk_number is None or self.chunk_length is None or self.overlap_seconds is None:
            return 0.0
        
        # Each chunk starts at: chunk_number * (chunk_length - overlap_seconds)
        # This accounts for the overlap between consecutive chunks
        chunk_step = self.chunk_length - self.overlap_seconds
        return self.chunk_number * chunk_step
    
    def __repr__(self):
        chunk_info = f" (chunk {self.chunk_number})" if self.chunk_number is not None else ""
        return f"AudioFile({self.title}{chunk_info})"

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def format_srt_time(srt_time: str) -> str:
    """Convert SRT time format '00:01:23,456' to '[0:01:23]' format."""
    # Remove milliseconds and convert to [H:MM:SS] or [MM:SS] format
    time_part = srt_time.split(',')[0]  # Remove milliseconds
    parts = time_part.split(':')
    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    
    if hours > 0:
        return f"[{hours}:{minutes:02d}:{seconds:02d}]"
    else:
        return f"[{minutes}:{seconds:02d}]"

def parse_srt_to_timestamped_text(srt_content: str, chunk_offset: float = 0.0) -> str:
    """
    Parse SRT content and return timestamped text in format: [0:01:23] text
    
    Args:
        srt_content: SRT format string from Whisper
        chunk_offset: Offset in seconds to add to timestamps for chunk alignment
    
    Returns:
        Formatted text with timestamps
    """
    blocks = srt_content.strip().split('\n\n')
    processed_lines = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # SRT format: index, time_range, text
            time_range = lines[1]
            text = '\n'.join(lines[2:])  # Handle multi-line text
            
            # Extract start time
            start_time_str = time_range.split(' --> ')[0]
            
            # If we have a chunk offset, we need to adjust the timestamp
            if chunk_offset > 0:
                # Convert SRT time to seconds, add offset, convert back
                time_parts = start_time_str.replace(',', '.').split(':')
                total_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])
                adjusted_seconds = total_seconds + chunk_offset
                
                # Convert back to SRT format for formatting
                hours = int(adjusted_seconds // 3600)
                minutes = int((adjusted_seconds % 3600) // 60)
                seconds = int(adjusted_seconds % 60)
                start_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d},000"
            
            # Format the timestamp
            formatted_time = format_srt_time(start_time_str)
            processed_line = f"{formatted_time} {text}"
            processed_lines.append(processed_line)
    
    return '\n'.join(processed_lines)

def split_audio_file(audio_path: str, chunk_length_seconds: int = 600, output_dir: str = None, overlap_seconds: int = 20) -> List[str]:
    """
    Split an audio file into smaller chunks with optional overlap.
    
    Args:
        audio_path: Path to the audio file
        chunk_length_seconds: Length of each chunk in seconds (default: 600 = 10 minutes)
        output_dir: Directory to save chunks (default: same as audio file)
        overlap_seconds: Seconds of overlap between consecutive chunks (default: 20)
    
    Returns:
        List of paths to the audio chunks
    """
    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get audio duration
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1', str(audio_path)],
            capture_output=True,
            text=True,
            check=True
        )
        total_duration = float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}. Not splitting.")
        return [str(audio_path)]
    
    # If audio is shorter than chunk length, don't split
    if total_duration <= chunk_length_seconds:
        logger.info(f"Audio duration ({total_duration:.0f}s) <= chunk length ({chunk_length_seconds}s), not splitting")
        return [str(audio_path)]
    
    logger.info(f"Splitting {audio_path.name} ({total_duration:.0f}s) into {chunk_length_seconds}s chunks with {overlap_seconds}s overlap")
    
    chunks = []
    chunk_step = chunk_length_seconds - overlap_seconds  # Step between chunk starts
    i = 0
    
    while True:
        start_time = i * chunk_step
        if start_time >= total_duration:
            break
        
        chunk_path = output_dir / f"{audio_path.stem}_chunk_{i:03d}{audio_path.suffix}"
        
        try:
            subprocess.run(
                ['ffmpeg', '-i', str(audio_path), '-ss', str(start_time),
                 '-t', str(chunk_length_seconds), '-c', 'copy', str(chunk_path),
                 '-y', '-loglevel', 'error'],
                check=True
            )
            chunks.append(str(chunk_path))
            logger.info(f"Created chunk {i}: {chunk_path.name} (start: {start_time:.0f}s)")
        except Exception as e:
            logger.error(f"Error creating chunk {i}: {e}")
        
        i += 1
    
    logger.info(f"Created {len(chunks)} chunks from {audio_path.name}")
    
    # Remove the original audio file after successful chunking
    try:
        audio_path.unlink()
        logger.info(f"Deleted original audio file: {audio_path.name}")
    except Exception as e:
        logger.warning(f"Could not delete original audio file: {e}")
    
    return chunks

@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio."""
    text: str
    start_time: float
    end_time: float
    audio_file: AudioFile
    chunk_index: int = 0  # Track which chunk this segment came from
    
    def __repr__(self):
        return f"Segment({self.start_time:.1f}s: {self.text[:50]}...)"

@dataclass
class TranscriptChunk:
    """
    Represents a fully transcribed audio chunk with all segments ready for entity extraction.
    
    This is the data structure that flows through the transcription-to-entity-extraction pipeline.
    It packages all segments from a single audio chunk together so they can be batch-processed
    for entity extraction. This improves performance compared to processing individual segments.
    
    Attributes:
        audio_file: Reference to the source AudioFile object (contains video_id, title, duration)
        segments: List of TranscriptSegment objects extracted from this chunk
        full_text: Concatenated text of all segments (useful for context)
    """
    audio_file: AudioFile
    segments: List['TranscriptSegment']
    full_text: str
    
    def __repr__(self):
        return f"TranscriptChunk({self.audio_file.title}, {len(self.segments)} segments)"

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: str
    audio_file: AudioFile
    timestamp_start: float
    timestamp_end: float
    context: str
    
    def to_dict(self):
        # Calculate YouTube links with timestamps
        start_seconds = int(self.timestamp_start)
        end_seconds = int(self.timestamp_end)
        
        # Main link starts at the exact timestamp where entity appears
        youtube_link = f"https://youtube.com/watch?v={self.audio_file.video_id}&t={start_seconds}s"
        
        # Alternative link with time range (some players support this)
        youtube_link_range = f"https://youtube.com/watch?v={self.audio_file.video_id}&t={start_seconds}s&end={end_seconds}s"
        
        return {
            'entity': self.text,
            'type': self.entity_type,
            'video_info': {
                'title': self.audio_file.title,
                'video_id': self.audio_file.video_id,
                'source_audio': self.audio_file.path,
                'youtube_link': youtube_link,
                'youtube_link_range': youtube_link_range,
            },
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'timestamp_start_formatted': self._format_timestamp(self.timestamp_start),
            'timestamp_end_formatted': self._format_timestamp(self.timestamp_end),
            'context': self.context,
            'extraction_time': datetime.now().isoformat()
        }
    
    def _format_timestamp(self, seconds):
        """Format timestamp as MM:SS or HH:MM:SS"""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

# =====================================================================
# PRODUCER: DOWNLOAD STAGE
# =====================================================================

class DownloadProducer(threading.Thread):
    """Producer thread that downloads videos and queues audio files."""
    
    def __init__(self, output_queue: queue.Queue, playlist_url: str, audio_dir: str, 
                 max_videos: Optional[int] = None, chunk_length: int = 600, 
                 extract_info: bool = True, cookies_from_browser: Optional[str] = None,
                 cookies_file: Optional[str] = None):
        super().__init__(name="DownloadProducer", daemon=False)
        self.output_queue = output_queue
        self.playlist_url = playlist_url
        self.audio_dir = Path(audio_dir)
        self.max_videos = max_videos
        self.chunk_length = chunk_length  # Split audio into chunks of this length (seconds)
        self.extract_info = extract_info  # Whether to extract new playlist info or use existing
        self.cookies_from_browser = cookies_from_browser
        self.cookies_file = cookies_file
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
    
    def _get_playlist_info_file(self):
        """Get the path to the playlist_info.json file."""
        output_dir = Path(self.audio_dir).parent
        return output_dir / 'playlist_info.json'
    
    def _load_playlist_info(self):
        """Load playlist info from JSON file."""
        playlist_file = self._get_playlist_info_file()
        if playlist_file.exists():
            with open(playlist_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_playlist_info(self, playlist_data):
        """Save playlist info to JSON file."""
        playlist_file = self._get_playlist_info_file()
        with open(playlist_file, 'w', encoding='utf-8') as f:
            json.dump(playlist_data, f, indent=2, ensure_ascii=False)
    
    def _mark_video_processed(self, video_id):
        """Mark a video as processed in the playlist_info.json file."""
        playlist_data = self._load_playlist_info()
        if playlist_data:
            for video in playlist_data['videos']:
                if video['id'] == video_id:
                    video['processed'] = True
                    logger.info(f"Marked video as processed: {video['title']} ({video_id})")
                    break
            self._save_playlist_info(playlist_data)
    
    def _get_ytdlp_options(self, base_options=None):
        """Generate yt-dlp options with cookie support."""
        if base_options is None:
            base_options = {}
        
        options = base_options.copy()
        
        # Add cookie support
        if self.cookies_from_browser:
            options['cookiesfrombrowser'] = (self.cookies_from_browser, None, None, None)
            logger.info(f"Using cookies from {self.cookies_from_browser} browser")
        elif self.cookies_file:
            options['cookiefile'] = self.cookies_file
            logger.info(f"Using cookies from file: {self.cookies_file}")
        
        return options
    
    def _get_next_unprocessed_videos(self, max_count=None):
        """Get the next unprocessed videos from the playlist."""
        playlist_data = self._load_playlist_info()
        if not playlist_data:
            return []
        
        unprocessed_videos = []
        for video in playlist_data['videos']:
            # Skip if already processed
            if video.get('processed', False):
                continue
            
            unprocessed_videos.append(video)
            
            # Stop if we have enough videos
            if max_count and len(unprocessed_videos) >= max_count:
                break
                
        return unprocessed_videos
    
    def _download_single_video(self, entry):
        """Download and process a single video entry."""
        base_download_opts = {
            # Optimized format selection for fast audio-only downloads:
            # 1. bestaudio[acodec^=mp4a]: MP4 audio-only streams (usually fastest)
            # 2. bestaudio[acodec^=opus]: Opus audio streams (good compression)  
            # 3. worst[height<=480]: Low quality video as fallback (faster than best)
            # 4. bestaudio: Any audio-only stream
            # 5. worst: Lowest quality video (last resort)
            'format': 'bestaudio[acodec^=mp4a]/bestaudio[acodec^=opus]/worst[height<=480]/bestaudio/worst',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',  # Reduced from 192 to 128 kbps for faster downloads
            }],
            'outtmpl': str(self.audio_dir / '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'no_playlist': True,  # Don't try to download as playlist
        }
        
        # Add cookie support
        download_opts = self._get_ytdlp_options(base_download_opts)

        video_id = entry.get('id')
        title = entry.get('title', 'Unknown')
        duration = entry.get('duration')

        logger.info(f"Downloading: {title}")

        # Download this specific video
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.extract_info(video_url, download=True)

        # Find the most recently downloaded audio file
        audio_files = sorted(self.audio_dir.glob('*.mp3'), key=lambda x: x.stat().st_mtime, reverse=True)
        if audio_files:
            audio_file = audio_files[0]

            # Split audio into chunks with 20-second overlap
            chunks = split_audio_file(str(audio_file), self.chunk_length, str(self.audio_dir), overlap_seconds=20)

            # Sort chunks by chunk number to ensure sequential processing
            def extract_chunk_number(chunk_path):
                # Extract chunk number from filename like "video_chunk_005.mp3"
                import re
                match = re.search(r'_chunk_(\d+)\.mp3$', str(chunk_path))
                return int(match.group(1)) if match else 0
            
            chunks_sorted = sorted(chunks, key=extract_chunk_number)

            # Queue each chunk in order
            for chunk_path in chunks_sorted:
                # Extract chunk number from filename
                chunk_number = extract_chunk_number(chunk_path)
                
                audio_obj = AudioFile(
                    path=chunk_path,
                    video_id=video_id,
                    title=title,
                    duration=duration,
                    chunk_number=chunk_number,
                    chunk_length=self.chunk_length,
                    overlap_seconds=20  # This should match the overlap used in split_audio_file
                )
                self.output_queue.put(audio_obj)
                logger.info(f"Queued: {audio_obj}")

    def _download_videos_continuously(self):
        """Continuously download unprocessed videos until none remain."""
        total_downloaded = 0
        
        while True:
            if self.stop_event.is_set():
                logger.info("Download producer stopped")
                break
            
            # Get next unprocessed video (one at a time for better flow control)
            unprocessed_videos = self._get_next_unprocessed_videos(max_count=1)
            
            if not unprocessed_videos:
                logger.info("No more unprocessed videos found - download producer finished")
                break
            
            entry = unprocessed_videos[0]
            
            try:
                # Convert to the format expected by download method
                download_entry = {
                    'id': entry['id'],
                    'title': entry['title'],
                    'url': entry['url'],
                    'duration': entry['duration']
                }
                
                self._download_single_video(download_entry)
                total_downloaded += 1
                
                logger.info(f"Completed video {total_downloaded}: {entry['title']}")
                
                # CRITICAL FIX: Mark successful video as processed to prevent infinite retry loop
                self._mark_video_processed(entry['id'])
                logger.info(f"Marked successful video as processed: {entry['title']}")
                
                # Brief pause to allow transcription to catch up and prevent overwhelming the system
                # This helps maintain good flow control between download and transcription stages
                time.sleep(2)
                
                # Check if we've reached max_videos limit
                if self.max_videos and total_downloaded >= self.max_videos:
                    logger.info(f"Reached max_videos limit ({self.max_videos})")
                    break

            except Exception as e:
                logger.warning(f"Skipping video {entry['id']} ({entry['title']}) due to download error: {e}")
                
                # CRITICAL FIX: Mark failed video as processed to avoid infinite retry loop
                self._mark_video_processed(entry['id'])
                logger.info(f"Marked failed video as processed to prevent retry: {entry['title']}")
                continue
        
        # Signal end of downloads
        self.output_queue.put(None)
        logger.info(f"Download producer finished - downloaded {total_downloaded} videos")
    
    def run(self):
        """Download videos from playlist and split into chunks."""
        logger.info(f"Starting download from: {self.playlist_url}")
        logger.info(f"Audio chunks will be {self.chunk_length}s (~{self.chunk_length/60:.0f} minutes) each")

        # Step 1: Check extract_info flag
        if not self.extract_info:
            logger.info("extract_info=False: Using existing playlist_info.json only")
            existing_playlist_data = self._load_playlist_info()
            if not existing_playlist_data:
                logger.error("No existing playlist_info.json found and extract_info=False. Cannot proceed.")
                logger.error("Either run with --info to extract playlist info, or ensure playlist_info.json exists.")
                self.output_queue.put(None)
                return
            
            # Use continuous downloading - will process ALL unprocessed videos
            logger.info("Starting continuous processing of unprocessed videos...")
            self._download_videos_continuously()
            return

        # Step 2: extract_info=True - Always extract fresh playlist information from the provided URL
        logger.info("extract_info=True: Extracting fresh playlist information from provided URL...")
        
        # Extract playlist metadata - get ALL videos from the NEW URL
        try:
            logger.info("Extracting playlist information...")
            base_extraction_opts = {
                'quiet': False,
                'no_warnings': False,
                'ignoreerrors': True,  # Skip unavailable/live videos during extraction
                'extract_flat': True,  # Get only basic info, don't extract full metadata
            }
            # Add cookie support for playlist extraction
            extraction_opts = self._get_ytdlp_options(base_extraction_opts)
            
            with yt_dlp.YoutubeDL(extraction_opts) as ydl:
                playlist_info = ydl.extract_info(self.playlist_url, download=False)
        except Exception as e:
            logger.error(f"Unrecoverable error extracting playlist info: {e}")
            self.output_queue.put(None)
            return

        # Handle case where playlist extraction returns None
        if playlist_info is None:
            logger.error("Failed to extract playlist information - playlist may not exist or be private")
            self.output_queue.put(None)
            return

        # Get playlist details
        playlist_title = playlist_info.get('title', 'Unknown Playlist')
        playlist_id = playlist_info.get('id', 'Unknown ID')
        all_entries = playlist_info.get('entries', [])
        
        # Handle single video case (when URL is not a playlist)
        if not all_entries and playlist_info.get('id'):
            # This is a single video, not a playlist
            all_entries = [playlist_info]
            playlist_title = f"Single Video: {playlist_info.get('title', 'Unknown')}"
            logger.info(f"Processing single video: {playlist_info.get('title', 'Unknown')}")

        # Filter out live/unavailable videos
        filtered_entries = []
        for entry in all_entries:
            # Skip None entries (yt-dlp can return None for truly unavailable videos)
            if entry is None:
                logger.warning("Skipping None entry (unavailable video)")
                continue
                
            # yt-dlp marks unavailable/live videos with 'is_live' or missing 'duration'
            if entry.get('is_live') or entry.get('duration') is None:
                logger.warning(f"Skipping unavailable/live video: {entry.get('title', 'Unknown')} ({entry.get('id')})")
                continue
            filtered_entries.append(entry)

        logger.info(f"Found {len(filtered_entries)} downloadable videos in playlist (out of {len(all_entries)} total)")

        # Save playlist info to file - only include downloadable videos
        playlist_data = {
            'title': playlist_title,
            'id': playlist_id,
            'url': self.playlist_url,
            'total_videos_found': len(all_entries),
            'available_videos': len(filtered_entries),
            'extracted_at': datetime.now().isoformat(),
            'videos': [
                {
                    'id': entry.get('id'),
                    'title': entry.get('title'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'duration': entry.get('duration'),
                    'processed': False,  # Track processing status
                }
                for entry in filtered_entries
            ]
        }

        # Save playlist info in output directory
        output_dir = Path(self.audio_dir).parent  # Go up one level to output dir
        playlist_info_file = output_dir / 'playlist_info.json'
        with open(playlist_info_file, 'w', encoding='utf-8') as f:
            json.dump(playlist_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved playlist info ({len(filtered_entries)} available videos) to: {playlist_info_file}")

        # Start continuous downloading - will process ALL videos or up to max_videos limit
        logger.info("Starting continuous processing of all videos...")
        self._download_videos_continuously()

# =====================================================================
# WORKER: TRANSCRIPTION STAGE
# =====================================================================

class TranscriptionWorker(threading.Thread):
    """Worker thread that transcribes audio files and writes transcripts immediately."""
    
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, 
                 transcripts_dir: str, model: str = 'small', language: str = 'es', device: str = 'auto'):
        super().__init__(name="TranscriptionWorker", daemon=False)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.language = language
        self.device = device if device != 'auto' else self._get_device()
        self.model = None
        self.stop_event = threading.Event()
        
        # Track video transcripts being built
        self.video_transcript_files = {}  # video_id -> file path
    
    def _get_device(self):
        """Auto-detect best device."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def _get_or_create_transcript_file(self, audio_file: AudioFile):
        """Get or create transcript file for this video."""
        video_id = audio_file.video_id
        
        if video_id not in self.video_transcript_files:
            # Create safe filename from title
            safe_title = "".join(c for c in audio_file.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filepath = self.transcripts_dir / f"{safe_title}_transcript_building.txt"
            self.video_transcript_files[video_id] = filepath
        
        return self.video_transcript_files[video_id]
    
    def run(self):
        """Transcribe audio files from queue and write transcripts immediately.
        
        CRITICAL DESIGN: This worker writes transcripts to file IMMEDIATELY after each
        chunk is transcribed. This enables real-time progress visibility instead of waiting
        for the entire video to be processed before seeing results.
        
        Data flow:
            1. Get AudioFile from input_queue
            2. Transcribe with Whisper -> get segments
            3. WRITE segments to disk immediately (append mode)
            4. Package segments into TranscriptChunk object
            5. PUT TranscriptChunk into output_queue for entity extraction
            6. Meanwhile, DownloadProducer can queue more chunks (parallelism!)
        """
        try:
            logger.info(f"Loading Whisper model: {self.model_name} (device: {self.device})")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully")
            
            while not self.stop_event.is_set():
                try:
                    # Get audio file from queue (timeout to check stop_event)
                    audio_file = self.input_queue.get(timeout=1)
                    
                    if audio_file is None:  # Sentinel value - no more files coming
                        logger.info("Transcription worker received end signal")
                        self.output_queue.put(None)  # Signal EntityConsumer to finish
                        break
                    
                    logger.info(f"Transcribing: {audio_file}")
                    
                    # Transcribe with Whisper
                    result = self.model.transcribe(
                        audio_file.path,
                        language=self.language,
                        verbose=False,  # Show real-time progress
                        fp16=False,  # Disable half-precision to avoid NaN issues
                    )
                    
                    segments = result.get('segments', [])
                    logger.info(f"Got {len(segments)} segments from transcription")
                    
                    # === KEY FEATURE: Write transcript to file IMMEDIATELY ===
                    # This enables real-time progress visibility and means we don't lose data
                    # if the process is interrupted during entity extraction.
                    
                    # Create SRT content from segments for timestamp formatting
                    srt_content = ""
                    for i, seg in enumerate(segments, 1):
                        start_time = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{int(seg['start']%60):02d},{int((seg['start']%1)*1000):03d}"
                        end_time = f"{int(seg['end']//3600):02d}:{int((seg['end']%3600)//60):02d}:{int(seg['end']%60):02d},{int((seg['end']%1)*1000):03d}"
                        srt_content += f"{i}\n{start_time} --> {end_time}\n{seg['text'].strip()}\n\n"
                    
                    # Parse SRT to timestamped text with chunk offset
                    chunk_offset = audio_file.get_time_offset()
                    timestamped_text = parse_srt_to_timestamped_text(srt_content, chunk_offset)
                    
                    transcript_file = self._get_or_create_transcript_file(audio_file)
                    
                    with open(transcript_file, 'a', encoding='utf-8') as f:
                        f.write(timestamped_text + '\n')
                        f.flush()  # Ensure data is written to disk immediately
                    
                    logger.info(f"✓ Wrote transcript chunk to: {transcript_file}")
                    
                    # Create TranscriptChunk object with all segments from this audio chunk
                    # Adjust timestamps to be relative to the complete video, not just this chunk
                    chunk_offset = audio_file.get_time_offset()
                    
                    transcript_chunk = TranscriptChunk(
                        audio_file=audio_file,
                        segments=[
                            TranscriptSegment(
                                text=seg['text'],
                                start_time=seg['start'] + chunk_offset,  # Add chunk offset
                                end_time=seg['end'] + chunk_offset,      # Add chunk offset
                                audio_file=audio_file,
                                chunk_index=0
                            )
                            for seg in segments
                        ],
                        full_text='\n'.join([seg['text'] for seg in segments])  # Plain text for entity extraction
                    )
                    
                    # Queue the complete chunk for entity extraction
                    # EntityConsumer can now start working on this while DownloadProducer
                    # is still downloading/processing other videos!
                    self.output_queue.put(transcript_chunk)
                    
                    # === CLEANUP: Delete the audio chunk after transcription ===
                    # This saves disk space since we no longer need the chunk file
                    try:
                        chunk_path = Path(audio_file.path)
                        if chunk_path.exists():
                            chunk_path.unlink()
                            logger.info(f"🗑️  Deleted audio chunk: {chunk_path.name}")
                        else:
                            logger.warning(f"⚠️  Audio chunk not found for deletion: {chunk_path}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to delete audio chunk {audio_file.path}: {e}")
                    
                    logger.info(f"Finished transcribing and queuing: {audio_file}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
            
            logger.info("Transcription worker finished")
            
        except Exception as e:
            logger.error(f"Transcription worker fatal error: {e}")
            self.output_queue.put(None)

# =====================================================================
# CONSUMER: ENTITY EXTRACTION STAGE
# =====================================================================

class EntityConsumer(threading.Thread):
    """Consumer thread that extracts entities from transcribed segments."""
    
    def __init__(self, input_queue: queue.Queue, output_dir: str, 
                 buffer_size: int = 50, spacy_model: str = 'es_core_news_lg',
                 video_completion_callback=None):
        super().__init__(name="EntityConsumer", daemon=False)
        self.input_queue = input_queue
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.video_completion_callback = video_completion_callback
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spaCy model with fallback
        try:
            logger.info(f"Loading spaCy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            try:
                logger.warning(f"{spacy_model} not found, falling back to es_core_news_md")
                self.nlp = spacy.load("es_core_news_md")
                logger.info("Loaded spaCy model: es_core_news_md")
            except OSError:
                logger.warning("es_core_news_md not found, falling back to es_core_news_sm")
                self.nlp = spacy.load("es_core_news_sm")
                logger.info("Loaded spaCy model: es_core_news_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None
        
        # Output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entities_jsonl = self.output_dir / f"entities_{timestamp}.jsonl"
        self.entities_summary = self.output_dir / f"entities_summary_{timestamp}.json"
        
        # Stats
        self.segment_count = 0
        self.entity_count = 0
        self.entity_database = defaultdict(list)
        self.entity_counts = defaultdict(Counter)
        
        # Track segments and entities by video
        self.video_segments = defaultdict(list)  # video_id -> list of segments
        self.video_entities = defaultdict(lambda: defaultdict(list))  # video_id -> entity_name -> list of entities
        
        self.stop_event = threading.Event()
        self.segment_buffer = []
    
    def run(self):
        """Extract entities from transcript chunks."""
        try:
            logger.info(f"Entity consumer started, writing to: {self.entities_jsonl}")
            
            while not self.stop_event.is_set():
                try:
                    # Get transcript chunk from queue
                    chunk = self.input_queue.get(timeout=1)
                    
                    if chunk is None:  # Sentinel value
                        logger.info("Entity consumer received end signal")
                        self._finalize()
                        break
                    
                    # Extract entities from all segments in this chunk
                    for segment in chunk.segments:
                        # Add to buffer
                        self.segment_buffer.append(segment)
                        self.segment_count += 1
                        
                        # Process buffer when full
                        if len(self.segment_buffer) >= self.buffer_size:
                            self._process_buffer()
                    
                    logger.info(f"✓ Queued {len(chunk.segments)} segments from: {chunk.audio_file.title}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Entity extraction error: {e}")
            
            logger.info("Entity consumer finished")
            
        except Exception as e:
            logger.error(f"Entity consumer fatal error: {e}")
    
    def _process_buffer(self):
        """
        Process buffered segments using batch entity extraction.
        
        OPTIMIZATION: Instead of extracting entities from each segment individually,
        we batch them together and process multiple segments at once with spaCy.
        This is much faster than processing segment-by-segment.
        
        Process:
            1. Collect segments into buffer (up to buffer_size)
            2. Extract text from all segments
            3. Batch process all texts with spaCy.pipe() (efficient!)
            4. For each (document, segment) pair:
               - Extract spaCy entities (PERSON, ORG, GPE, MISC)
               - Extract custom pattern entities (POLITICAL_TITLES, ORGANIZATIONS, etc)
               - Write each entity to JSONL immediately
               - Track by video_id for per-video aggregation later
            5. Clear buffer and repeat
        """
        if not self.segment_buffer or not self.nlp:
            return
        
        batch_texts = [seg.text for seg in self.segment_buffer]
        
        # Batch process with spaCy - much faster than sequential processing
        docs = list(self.nlp.pipe(batch_texts, batch_size=25, disable=['parser', 'tagger']))
        
        for doc, segment in zip(docs, self.segment_buffer):
            # Track segment by video for later per-video file generation
            self.video_segments[segment.audio_file.video_id].append(segment)
            
            # ===== EXTRACT SPACY NAMED ENTITIES =====
            for ent in doc.ents:
                # Accept more entity types and lower length threshold for names
                valid_types = ['PERSON', 'ORG', 'GPE', 'MISC', 'LOC', 'EVENT', 'FAC', 'NORP']
                min_length = 2 if ent.label_ == 'PERSON' else 3  # Lower threshold for person names
                
                if len(ent.text.strip()) >= min_length and ent.label_ in valid_types:
                    entity_text = ent.text.strip()
                    
                    # Skip common words that are incorrectly tagged as entities
                    skip_words = {'este', 'esta', 'esa', 'ese', 'esto', 'eso', 'aquí', 'ahí', 'allí', 
                                 'sí', 'no', 'que', 'como', 'cuando', 'donde', 'porque', 'pero',
                                 'más', 'muy', 'también', 'ahora', 'después', 'antes', 'bien', 'mal'}
                    
                    if entity_text.lower() not in skip_words:
                        self.entity_counts[ent.label_][entity_text] += 1
                        
                        # Calculate real timestamps relative to the complete video
                        # NOTE: segment.start_time and segment.end_time already include chunk_offset
                        # from the TranscriptionWorker, so don't add it again!
                        real_start_time = segment.start_time
                        real_end_time = segment.end_time
                        
                        entity_obj = Entity(
                            text=entity_text,
                            entity_type=ent.label_,
                            audio_file=segment.audio_file,
                            timestamp_start=real_start_time,
                            timestamp_end=real_end_time,
                            context=segment.text
                        )
                        
                        # Store in databases
                        self.entity_database[entity_text].append(entity_obj.to_dict())
                        self.video_entities[segment.audio_file.video_id][entity_text].append(entity_obj.to_dict())
                        
                        # Write to JSONL immediately for real-time visibility
                        self._write_entity(entity_obj)
                        self.entity_count += 1
        
        self.segment_buffer = []
        logger.info(f"Processed buffer: {self.segment_count} segments, {self.entity_count} total entities")
    
    def _write_entity(self, entity: Entity):
        """Write entity to JSONL file."""
        try:
            with open(self.entities_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entity.to_dict(), ensure_ascii=False) + '\n')
                f.flush()
        except Exception as e:
            logger.error(f"Error writing entity: {e}")
    
    def _finalize(self):
        """
        Finalize and save all results.
        
        This is called when EntityConsumer receives the sentinel (None) value,
        indicating no more data will arrive. It:
        
        1. Processes any remaining buffered segments
        2. Saves global summary (total counts, timestamps, etc)
        3. Creates per-video files:
           - _transcript.txt: Raw transcript text
           - _transcript.json: Structured segments with timestamps
           - _entities.json: Entities found in that specific video
        
        These files are saved to output_dir/video_transcripts/ for easy browsing.
        """
        # Process remaining buffer
        if self.segment_buffer:
            self._process_buffer()
        
        # ===== SAVE GLOBAL SUMMARY =====
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_segments_processed': self.segment_count,
            'total_entities_found': self.entity_count,
            'entity_counts': {k: dict(v) for k, v in self.entity_counts.items()},
            'jsonl_file': str(self.entities_jsonl),
        }
        
        with open(self.entities_summary, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary to: {self.entities_summary}")
        
        # ===== SAVE PER-VIDEO TRANSCRIPTS AND ENTITIES =====
        logger.info(f"Saving per-video transcripts and entities...")
        
        video_transcripts_dir = self.output_dir / 'video_transcripts'
        video_transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        for video_id, segments in self.video_segments.items():
            if not segments:
                continue
            
            # Find video title from first segment
            video_title = segments[0].audio_file.title if segments else f"video_{video_id}"
            safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # ===== SAVE TRANSCRIPT FILES =====
            # 1. Plain text transcript with timestamps
            transcript_file = video_transcripts_dir / f"{safe_title}_transcript.txt"
            
            # Create timestamped text from segments
            timestamped_lines = []
            for seg in segments:
                # Format timestamp like [0:01:23]
                start_seconds = int(seg.start_time)
                hours = start_seconds // 3600
                minutes = (start_seconds % 3600) // 60
                seconds = start_seconds % 60
                
                if hours > 0:
                    timestamp = f"[{hours}:{minutes:02d}:{seconds:02d}]"
                else:
                    timestamp = f"[{minutes}:{seconds:02d}]"
                
                timestamped_lines.append(f"{timestamp} {seg.text}")
            
            full_text_with_timestamps = '\n'.join(timestamped_lines)
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(full_text_with_timestamps)
            
            # 2. Structured transcript with timestamps
            transcript_json = video_transcripts_dir / f"{safe_title}_transcript.json"
            transcript_data = {
                'video_id': video_id,
                'video_title': video_title,
                'total_segments': len(segments),
                'text': '\n'.join([seg.text for seg in segments]),  # Plain text without timestamps
                'text_with_timestamps': full_text_with_timestamps,  # Text with timestamps
                'segments': [
                    {
                        'start': seg.start_time,
                        'end': seg.end_time,
                        'text': seg.text
                    }
                    for seg in segments
                ],
                'created_at': datetime.now().isoformat()
            }
            
            with open(transcript_json, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved transcript for {video_title} to: {transcript_file}")
            
            # ===== SAVE ENTITIES FOR THIS VIDEO =====
            # Aggregate all entities found in this video
            entities_file = video_transcripts_dir / f"{safe_title}_entities.json"
            
            video_entity_data = {
                'video_id': video_id,
                'video_title': video_title,
                'total_entities': len(self.video_entities.get(video_id, {})),
                'entities': {}
            }
            
            for entity_name, entity_list in self.video_entities.get(video_id, {}).items():
                video_entity_data['entities'][entity_name] = {
                    'type': entity_list[0].get('type') if entity_list else 'UNKNOWN',
                    'mentions': len(entity_list),
                    'occurrences': entity_list
                }
            
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(video_entity_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved entities for {video_title} to: {entities_file}")
            
            # Mark video as processed in playlist_info.json
            if self.video_completion_callback:
                self.video_completion_callback(video_id)

# =====================================================================
# PIPELINE ORCHESTRATOR
# =====================================================================

class Pipeline:
    """Orchestrates the producer-consumer pipeline."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / 'audio'
        self.transcripts_dir = self.output_dir / 'transcripts'
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, playlist_url: str, max_videos: Optional[int] = None,
            whisper_model: str = 'small', language: str = 'es',
            device: str = 'auto', buffer_size: int = 50, chunk_length: int = 600,
            extract_info: bool = True, cookies_from_browser: Optional[str] = None,
            cookies_file: Optional[str] = None):
        """Run the full pipeline."""
        
        logger.info("=" * 80)
        logger.info("Starting Aimara Pipeline")
        logger.info("=" * 80)
        
        # Create queues with flow control
        # download_queue: Limit to prevent excessive downloading ahead of transcription
        # Small size forces DownloadProducer to wait when TranscriptionWorker is busy
        download_queue = queue.Queue(maxsize=3)  # ~1-2 videos worth of chunks max
        
        # transcription_queue: Smaller size since EntityConsumer is very fast
        # Large buffer here would just consume memory unnecessarily
        transcription_queue = queue.Queue(maxsize=10)  # Enough to keep EntityConsumer busy
        
        # Create threads
        downloader = DownloadProducer(
            download_queue, 
            playlist_url, 
            str(self.audio_dir),
            max_videos=max_videos,
            chunk_length=chunk_length,
            extract_info=extract_info,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file
        )
        
        transcriber = TranscriptionWorker(
            download_queue,
            transcription_queue,
            transcripts_dir=str(self.transcripts_dir),
            model=whisper_model,
            language=language,
            device=device
        )
        
        extractor = EntityConsumer(
            transcription_queue,
            str(self.transcripts_dir),
            buffer_size=10,  # Reduced from default 50 to prevent memory issues
            video_completion_callback=downloader._mark_video_processed
        )
        
        # Start threads
        downloader.start()
        transcriber.start()
        extractor.start()
        
        # Wait for completion
        downloader.join()
        logger.info("Download stage completed")
        
        transcriber.join()
        logger.info("Transcription stage completed")
        
        extractor.join()
        logger.info("Entity extraction stage completed")
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Global entities saved to: {extractor.entities_jsonl}")
        logger.info(f"Global summary saved to: {extractor.entities_summary}")
        logger.info(f"Per-video transcripts and entities: {extractor.output_dir}/video_transcripts/")
        logger.info("=" * 80)
        
        return {
            'entities_file': str(extractor.entities_jsonl),
            'summary_file': str(extractor.entities_summary),
            'video_transcripts_dir': str(extractor.output_dir / 'video_transcripts'),
            'total_entities': extractor.entity_count,
            'total_segments': extractor.segment_count,
            'total_videos': len(extractor.video_segments),
        }

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        playlist_url = sys.argv[1]
        pipeline = Pipeline()
        pipeline.run(playlist_url, max_videos=2)
