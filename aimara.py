#!/usr/bin/env python3
"""
Aimara - YouTube Playlist Audio Processing and Transcription Tool
================================================================

A streamlined command-line utility for processing YouTube content:
- Fetching YouTube playlist information
- Downloading audio from videos  
- Full pipeline: Download → Transcribe → Extract Entities (recommended)
- Searching extracted entities and transcripts

Main Commands:
    aimara playlist <playlist_url>                    # Get playlist info
    aimara download <video_url>                       # Download single video audio
    aimara pipeline-process <playlist_url>            # Full pipeline (recommended)
    aimara search <query>                             # Search entities
    aimara status                                     # Show file status
    aimara cleanup                                    # Clean up files

The pipeline-process command is the main feature - it downloads, transcribes,
and extracts entities from playlists using a multi-threaded architecture.
"""

import click
import yt_dlp
from pathlib import Path
import json
import sys
from datetime import datetime

# Version
__version__ = "1.0.0"

# Global configuration
DEFAULT_OUTPUT_DIR = "."  # Current directory, not "output"
DEFAULT_AUDIO_DIR = "audio"
DEFAULT_TRANSCRIPTS_DIR = "transcripts"

class AimaraConfig:
    """Configuration management for Aimara."""
    
    def __init__(self):
        self.output_dir = Path(DEFAULT_OUTPUT_DIR)
        self.audio_dir = Path(DEFAULT_AUDIO_DIR)
        self.transcripts_dir = Path(DEFAULT_TRANSCRIPTS_DIR)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)

config = AimaraConfig()

def progress_hook(d):
    """Progress hook for yt-dlp."""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A').strip()
        speed = d.get('_speed_str', 'N/A').strip()
        eta = d.get('_eta_str', 'N/A').strip()
        click.echo(f"\r   Progress: {percent} at {speed} ETA: {eta}", nl=False)
    elif d['status'] == 'finished':
        click.echo("\n   ✅ Download finished, processing...")

# =====================================================================
# PLAYLIST COMMANDS
# =====================================================================

@click.group()
@click.version_option(version=__version__)
@click.option('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for all files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def cli(output_dir, verbose):
    """
    Aimara - YouTube Playlist Processing Pipeline
    
    🚀 Main command: pipeline-process PLAYLIST_URL
    
    Downloads videos, transcribes audio, and extracts entities in parallel.
    """
    config.output_dir = Path(output_dir)
    config.audio_dir = config.output_dir / "audio"
    config.transcripts_dir = config.output_dir / "transcripts"
    
    # Create directories
    config.output_dir.mkdir(exist_ok=True)
    config.audio_dir.mkdir(exist_ok=True)
    config.transcripts_dir.mkdir(exist_ok=True)
    
    if verbose:
        click.echo(f"📁 Output directory: {config.output_dir.resolve()}")

@cli.command()
@click.argument('playlist_url')
@click.option('--save-json/--no-save-json', default=True, help='Save playlist info to JSON file')
def playlist(playlist_url, save_json):
    """Fetch and display YouTube playlist information."""
    
    click.echo("🎵 Fetching playlist information...")
    click.echo(f"URL: {playlist_url}")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
    except Exception as e:
        click.echo(f"❌ Error fetching playlist: {e}")
        sys.exit(1)
    
    if not info:
        click.echo("❌ No playlist information found")
        sys.exit(1)
    
    # Display playlist info
    playlist_title = info.get('title', 'Unknown Playlist')
    playlist_id = info.get('id', 'Unknown ID')
    entries = info.get('entries', [])
    
    click.echo("=" * 80)
    click.echo(f"📺 Playlist: {playlist_title}")
    click.echo(f"📋 Playlist ID: {playlist_id}")
    click.echo(f"📊 Videos: {len(entries)}")
    click.echo("=" * 80)
    
    if entries:
        click.echo(f"\n✅ Found {len(entries)} videos:")
        for idx, entry in enumerate(entries[:10], 1):  # Show first 10
            video_id = entry.get('id', 'Unknown')
            video_title = entry.get('title', 'Unknown Title')
            duration = entry.get('duration', 'Unknown')
            
            duration_str = f" ({duration}s)" if duration else ""
            click.echo(f"  {idx:2d}. {video_title}{duration_str}")
            click.echo(f"      https://www.youtube.com/watch?v={video_id}")
        
        if len(entries) > 10:
            click.echo(f"      ... and {len(entries) - 10} more videos")
    
    # Save to JSON if requested
    if save_json:
        output_file = config.output_dir / "playlist_info.json"
        playlist_data = {
            'title': playlist_title,
            'id': playlist_id,
            'url': playlist_url,
            'video_count': len(entries),
            'fetched_at': datetime.now().isoformat(),
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
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(playlist_data, f, indent=2, ensure_ascii=False)
        
        click.echo(f"\n✅ Playlist info saved to: {output_file}")

# =====================================================================
# DOWNLOAD COMMANDS
# =====================================================================

@cli.command()
@click.argument('video_url')
@click.option('--quality', default='192', help='Audio quality (kbps)')
@click.option('--format', 'audio_format', default='mp3', help='Audio format (mp3, wav, etc.)')
def download(video_url, quality, audio_format):
    """Download audio from a YouTube video."""
    
    click.echo("🎵 Downloading audio...")
    click.echo(f"URL: {video_url}")
    click.echo(f"Quality: {quality} kbps")
    click.echo(f"Format: {audio_format}")
    click.echo(f"Destination: {config.audio_dir.resolve()}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': quality,
        }],
        'outtmpl': str(config.audio_dir / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
            # Find the generated audio file
            audio_files = list(config.audio_dir.glob(f'*.{audio_format}'))
            
            if audio_files:
                audio_file = audio_files[-1]  # Most recent file
                file_size_mb = audio_file.stat().st_size / (1024*1024)
                
                # Clean up any leftover webm files
                for webm_file in config.audio_dir.glob('*.webm'):
                    webm_file.unlink()
                
                click.echo(f"\n✅ Download complete!")
                click.echo(f"📁 Saved to: {audio_file.resolve()}")
                click.echo(f"📊 Size: {file_size_mb:.2f} MB")
                
                return str(audio_file)
            else:
                click.echo("❌ Audio file not found after download")
                sys.exit(1)
                
    except Exception as e:
        click.echo(f"❌ Error downloading audio: {e}")
        sys.exit(1)

# =====================================================================
# SEARCH COMMANDS
# =====================================================================

@cli.command()
@click.argument('query')
@click.option('--entity-type', help='Filter by entity type (PERSON, ORG, GPE, etc.)')
@click.option('--min-mentions', default=1, help='Minimum number of mentions')
@click.option('--show-context/--no-context', default=True, help='Show context around mentions')
def search(query, entity_type, min_mentions, show_context):
    """Search for entities or text in transcripts."""
    
    click.echo(f"🔍 Searching for: '{query}'")
    if entity_type:
        click.echo(f"   Entity type filter: {entity_type}")
    click.echo(f"   Minimum mentions: {min_mentions}")
    
    # Find all entity files (both old format and new JSONL format)
    entity_files = list(config.transcripts_dir.glob("entities_*.json"))
    jsonl_files = list(config.transcripts_dir.glob("entities_*.jsonl"))
    video_entity_files = list((config.transcripts_dir / "video_transcripts").glob("*_entities.json"))
    
    if not entity_files and not jsonl_files and not video_entity_files:
        click.echo("❌ No entity files found. Run the pipeline first:")
        click.echo("   uv run aimara.py pipeline-process PLAYLIST_URL")
        return
    
    results = []
    total_files = 0
    
    # Search video-specific entity files (new format)
    for entity_file in video_entity_files:
        try:
            with open(entity_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_files += 1
                
                video_title = data.get('video_title', 'Unknown')
                entities = data.get('entities', {})
                
                for entity_name, entity_data in entities.items():
                    if (query.lower() in entity_name.lower() and 
                        entity_data.get('mentions', 0) >= min_mentions):
                        
                        if not entity_type or entity_data.get('type') == entity_type:
                            for occurrence in entity_data.get('occurrences', []):
                                results.append({
                                    'entity': entity_name,
                                    'type': entity_data.get('type'),
                                    'video': video_title,
                                    'timestamp': occurrence.get('timestamp_start'),
                                    'context': occurrence.get('context', ''),
                                    'file': entity_file.name
                                })
        except Exception as e:
            click.echo(f"⚠️  Error reading {entity_file.name}: {e}")
    
    click.echo(f"\n📊 Searched {total_files} files, found {len(results)} matches")
    
    if results:
        # Group by entity
        from collections import defaultdict
        grouped = defaultdict(list)
        for result in results:
            grouped[result['entity']].append(result)
        
        click.echo(f"\n🎯 Results:")
        for entity, matches in sorted(grouped.items()):
            if len(matches) >= min_mentions:
                click.echo(f"\n🔹 {entity} ({matches[0]['type']}) - {len(matches)} mentions")
                
                for match in matches[:5]:  # Show first 5 matches
                    timestamp = match['timestamp']
                    video = match['video'][:50] + "..." if len(match['video']) > 50 else match['video']
                    
                    click.echo(f"   📹 {video}")
                    if timestamp:
                        click.echo(f"   ⏰ {timestamp:.1f}s")
                    
                    if show_context and match['context']:
                        context = match['context'][:100] + "..." if len(match['context']) > 100 else match['context']
                        click.echo(f"   💬 {context}")
                    click.echo()
                
                if len(matches) > 5:
                    click.echo(f"   ... and {len(matches) - 5} more matches")
    else:
        click.echo("❌ No matches found")

# =====================================================================
# UTILITY COMMANDS
# =====================================================================

@cli.command()
def status():
    """Show status of output directories and files."""
    
    click.echo("📊 Aimara Status")
    click.echo("=" * 50)
    click.echo(f"📁 Output directory: {config.output_dir.resolve()}")
    click.echo(f"🎵 Audio directory: {config.audio_dir.resolve()}")
    click.echo(f"📝 Transcripts directory: {config.transcripts_dir.resolve()}")
    
    # Count files
    audio_files = list(config.audio_dir.glob("*"))
    video_transcripts_dir = config.transcripts_dir / "video_transcripts"
    
    # New pipeline format files
    entity_jsonl_files = list(config.transcripts_dir.glob("entities_*.jsonl"))
    summary_files = list(config.transcripts_dir.glob("entities_summary_*.json"))
    video_transcript_files = list(video_transcripts_dir.glob("*_transcript.txt")) if video_transcripts_dir.exists() else []
    video_entity_files = list(video_transcripts_dir.glob("*_entities.json")) if video_transcripts_dir.exists() else []
    
    click.echo(f"\n📈 File counts:")
    click.echo(f"   🎵 Audio files: {len([f for f in audio_files if f.is_file()])}")
    click.echo(f"   � Video transcripts: {len(video_transcript_files)}")
    click.echo(f"   🔍 Video entity files: {len(video_entity_files)}")
    click.echo(f"   📊 Entity JSONL files: {len(entity_jsonl_files)}")
    click.echo(f"   📋 Summary files: {len(summary_files)}")
    
    if audio_files:
        total_size = sum(f.stat().st_size for f in audio_files if f.is_file())
        click.echo(f"   📊 Total audio size: {total_size / (1024*1024):.2f} MB")
    
    # Show recent pipeline runs
    if summary_files:
        click.echo(f"\n🕒 Recent pipeline runs:")
        for summary_file in sorted(summary_files, reverse=True)[:3]:
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    timestamp = data.get('timestamp', 'Unknown')
                    total_entities = data.get('total_entities_found', 0)
                    total_segments = data.get('total_segments_processed', 0)
                    click.echo(f"   • {timestamp}: {total_entities} entities, {total_segments} segments")
            except:
                click.echo(f"   • {summary_file.name}: (error reading)")
    
    # Quick pipeline usage tip
    if not entity_jsonl_files and not video_transcript_files:
        click.echo(f"\n💡 No pipeline output found. To get started:")
        click.echo(f"   uv run aimara.py pipeline-process PLAYLIST_URL --max-videos 1")

@cli.command()
@click.option('--keep-audio/--no-keep-audio', default=False, help='Keep audio files')
@click.option('--keep-transcripts/--no-keep-transcripts', default=False, help='Keep transcript files')
@click.confirmation_option(prompt='Are you sure you want to clean up files?')
def cleanup(keep_audio, keep_transcripts):
    """Clean up generated files."""
    
    removed_count = 0
    
    if not keep_audio:
        for audio_file in config.audio_dir.glob("*"):
            if audio_file.is_file():
                audio_file.unlink()
                removed_count += 1
        click.echo(f"🗑️  Removed {removed_count} audio files")
    
    removed_count = 0
    if not keep_transcripts:
        for transcript_file in config.transcripts_dir.glob("*"):
            if transcript_file.is_file():
                transcript_file.unlink()
                removed_count += 1
        click.echo(f"🗑️  Removed {removed_count} transcript files")
    
    click.echo("✅ Cleanup completed")

# =====================================================================
# PIPELINE COMMANDS
# =====================================================================

@cli.command()
@click.argument('playlist_url')
@click.option('--max-videos', default=None, type=int, help='Maximum number of videos to process')
@click.option('--model', default='small', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), help='Whisper model size')
@click.option('--language', default='es', help='Audio language (es, en, etc.)')
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps', 'auto']), default='auto', help='Processing device')
@click.option('--buffer-size', default=50, help='Entity extraction buffer size')
@click.option('--chunk-length', default=600, type=int, help='Audio chunk length in seconds (default: 600 = 10 minutes)')
def pipeline_process(playlist_url, max_videos, model, language, device, buffer_size, chunk_length):
    """
    🚀 MAIN COMMAND: Process YouTube playlist through complete pipeline.
    
    This is the recommended way to use Aimara. It:
    1. Downloads videos from playlist
    2. Splits long audio into chunks for faster processing  
    3. Transcribes with Whisper (parallel processing)
    4. Extracts entities in real-time
    5. Creates organized output files
    
    Multi-threaded producer-consumer architecture ensures maximum efficiency.
    Results are written to files as they're processed (real-time progress).
    
    Example:
        uv run aimara.py pipeline-process "https://youtube.com/playlist?list=..." --max-videos 2
    """
    
    click.echo("=" * 80)
    click.echo("🚀 Aimara Pipeline - Producer-Consumer Architecture")
    click.echo("=" * 80)
    click.echo(f"📺 Playlist: {playlist_url}")
    click.echo(f"📊 Max videos: {max_videos or 'unlimited'}")
    click.echo(f"🤖 Whisper model: {model}")
    click.echo(f"🌐 Language: {language}")
    click.echo(f"💻 Device: {device}")
    click.echo(f"📦 Buffer size: {buffer_size}")
    click.echo(f"✂️  Chunk length: {chunk_length}s (~{chunk_length/60:.0f} minutes)")
    click.echo("=" * 80)
    
    try:
        # Import pipeline here to avoid circular imports
        from pipeline import Pipeline
        
        # Create and run pipeline
        pipeline = Pipeline(output_dir=str(config.output_dir))
        result = pipeline.run(
            playlist_url=playlist_url,
            max_videos=max_videos,
            whisper_model=model,
            language=language,
            device=device,
            buffer_size=buffer_size,
            chunk_length=chunk_length
        )
        
        click.echo("\n✅ Pipeline completed successfully!")
        click.echo(f"\n📊 Results:")
        click.echo(f"   🎬 Total segments processed: {result['total_segments']}")
        click.echo(f"   🎥 Total videos processed: {result['total_videos']}")
        click.echo(f"   🔍 Total entities extracted: {result['total_entities']}")
        click.echo(f"\n📁 Output files:")
        click.echo(f"   � Global entities (JSONL): {result['entities_file']}")
        click.echo(f"   📊 Global summary: {result['summary_file']}")
        click.echo(f"   📂 Per-video files: {result['video_transcripts_dir']}")
        click.echo(f"      ├─ VIDEO_transcript.txt (full transcript)")
        click.echo(f"      ├─ VIDEO_transcript.json (segmented transcript)")
        click.echo(f"      └─ VIDEO_entities.json (entities for that video)")
        
    except Exception as e:
        click.echo(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    cli()