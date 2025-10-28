#!/usr/bin/env python3
"""Fetch and display all videos from a YouTube playlist."""

import yt_dlp
import json
from pathlib import Path

def get_playlist_videos(playlist_url):
    """
    Fetch all videos from a YouTube playlist.
    
    Args:
        playlist_url: Full YouTube playlist URL
        
    Returns:
        List of video information dictionaries
    """
    
    ydl_opts = {
        'quiet': False,
        'no_warnings': False,
        'extract_flat': 'in_playlist',  # Don't download, just extract playlist info
    }
    
    print(f"\n🎵 Fetching playlist information...")
    print(f"URL: {playlist_url}\n")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
        except Exception as e:
            print(f"❌ Error fetching playlist: {e}")
            return None
    
    return info

def display_playlist_info(playlist_info):
    """Display playlist information in a readable format."""
    
    if not playlist_info:
        return
    
    # Get playlist metadata
    playlist_title = playlist_info.get('title', 'Unknown Playlist')
    playlist_id = playlist_info.get('id', 'Unknown ID')
    
    print("=" * 80)
    print(f"📺 Playlist: {playlist_title}")
    print(f"📋 Playlist ID: {playlist_id}")
    print("=" * 80)
    
    # Get entries (videos)
    entries = playlist_info.get('entries', [])
    
    if not entries:
        print("❌ No videos found in playlist")
        return
    
    print(f"\n✅ Found {len(entries)} videos in playlist:\n")
    
    for idx, entry in enumerate(entries, 1):
        video_id = entry.get('id', 'Unknown')
        video_title = entry.get('title', 'Unknown Title')
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        duration = entry.get('duration', 'Unknown')
        
        print(f"{idx}. {video_title}")
        print(f"   ID: {video_id}")
        print(f"   URL: {video_url}")
        if duration:
            print(f"   Duration: {duration}s")
        print()
    
    print("=" * 80)
    print(f"Total Videos: {len(entries)}")
    print("=" * 80)
    
    # Save to JSON file for reference
    output_file = Path("playlist_info.json")
    with open(output_file, 'w') as f:
        json.dump({
            'title': playlist_title,
            'id': playlist_id,
            'video_count': len(entries),
            'videos': [
                {
                    'index': idx,
                    'id': entry.get('id'),
                    'title': entry.get('title'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'duration': entry.get('duration'),
                }
                for idx, entry in enumerate(entries, 1)
            ]
        }, f, indent=2)
    
    print(f"\n✅ Playlist info saved to: {output_file}")

if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/watch?v=4gHQ872IlkI&list=PLBN86rBUyEYD_2QDSC_-dXlZLVEdB7lp4"
    
    playlist_info = get_playlist_videos(playlist_url)
    display_playlist_info(playlist_info)
