#!/usr/bin/env python3
"""Download a single video from YouTube using yt-dlp."""

import yt_dlp
from pathlib import Path
import sys

def download_video(video_url, output_dir="downloads"):
    """
    Download a single video from YouTube.
    
    Args:
        video_url: YouTube video URL
        output_dir: Directory to save videos
        
    Returns:
        True if successful, False otherwise
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Best quality MP4 or fallback to best
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
    }
    
    print(f"\n🎬 Downloading video...")
    print(f"URL: {video_url}")
    print(f"Destination: {output_path.resolve()}\n")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            filepath = Path(filename)
            
            print(f"\n✅ Download complete!")
            print(f"📁 Saved to: {filepath.resolve()}")
            print(f"📊 Size: {filepath.stat().st_size / (1024*1024):.2f} MB")
            
            return True
            
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return False

def progress_hook(d):
    """Progress hook for yt-dlp."""
    if d['status'] == 'downloading':
        percent = d['_percent_str'].strip()
        speed = d['_speed_str'].strip()
        eta = d['_eta_str'].strip()
        print(f"   Progress: {percent} at {speed} ETA: {eta}", end='\r')
    elif d['status'] == 'finished':
        print("\n   Download finished, now post-processing...", end='\r')

if __name__ == "__main__":
    # Download second video from the playlist (Programa 548)
    video_url = "https://www.youtube.com/watch?v=P7zdykP0MWA"
    
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    
    success = download_video(video_url)
    sys.exit(0 if success else 1)
