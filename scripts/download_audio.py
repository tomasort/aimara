#!/usr/bin/env python3
"""Download audio only from a YouTube video using yt-dlp."""

import yt_dlp
from pathlib import Path
import sys

def download_audio(video_url, output_dir="audio"):
    """
    Download audio only from a YouTube video.
    
    Args:
        video_url: YouTube video URL
        output_dir: Directory to save audio files
        
    Returns:
        True if successful, False otherwise
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Configure yt-dlp options to extract audio only
    ydl_opts = {
        'format': 'bestaudio/best',  # Best audio quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',  # 192 kbps quality
        }],
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
    }
    
    print(f"\n🎵 Downloading audio only...")
    print(f"URL: {video_url}")
    print(f"Destination: {output_path.resolve()}\n")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
            # Find the generated audio file
            audio_files = list(output_path.glob('*.mp3'))
            
            if audio_files:
                audio_file = audio_files[-1]  # Most recent file
                file_size_mb = audio_file.stat().st_size / (1024*1024)
                
                # Clean up the original WebM/WEBM file
                for webm_file in output_path.glob('*.webm'):
                    webm_file.unlink()
                    print(f"   Cleaned up: {webm_file.name}")
                
                print(f"\n✅ Download complete!")
                print(f"📁 Saved to: {audio_file.resolve()}")
                print(f"📊 Size: {file_size_mb:.2f} MB")
                
                return True
            else:
                print("❌ Audio file not found after download")
                return False
            
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return False

def progress_hook(d):
    """Progress hook for yt-dlp."""
    if d['status'] == 'downloading':
        percent = d['_percent_str'].strip()
        speed = d['_speed_str'].strip()
        eta = d['_eta_str'].strip()
        print(f"   Progress: {percent} at {speed} ETA: {eta}", end='\r')
    elif d['status'] == 'finished':
        print("\n   Download finished, converting to MP3...", end='\r')

if __name__ == "__main__":
    # Download second video from the playlist (Programa 548)
    # This is about ~4.3 hours, so just a small portion
    video_url = "https://www.youtube.com/watch?v=P7zdykP0MWA"
    
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    
    success = download_audio(video_url)
    sys.exit(0 if success else 1)
