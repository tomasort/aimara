#!/usr/bin/env python3
"""Monitor real-time entity extraction from JSONL file."""

import json
import sys
import time
from pathlib import Path
from collections import Counter

def monitor_entities(filepath, follow=True):
    """Monitor and display entities as they're written to the real-time file."""
    
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"🔍 Monitoring entity extraction: {path.name}")
    print("=" * 60)
    
    entity_counts = Counter()
    total_entities = 0
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line:
                    try:
                        entry = json.loads(line.strip())
                        
                        if entry.get('event') == 'extraction_started':
                            print(f"🚀 Extraction started with buffer size: {entry.get('buffer_size')}")
                            print(f"   Time: {entry.get('timestamp')}")
                            print()
                        
                        elif entry.get('event') == 'extraction_completed':
                            print("\n✅ Extraction completed!")
                            stats = entry.get('final_stats', {})
                            print(f"   Total segments: {stats.get('segments_processed', 0)}")
                            print(f"   Total entities: {stats.get('total_entities', 0)}")
                            print(f"   Processing time: {stats.get('processing_time', 0):.2f}s")
                            
                            top_entities = entry.get('top_entities', {})
                            if top_entities:
                                print("\n🏆 Final top entities:")
                                for category, entities in top_entities.items():
                                    if entities:
                                        print(f"   {category}:")
                                        for name, count in entities:
                                            print(f"     • {name} ({count})")
                            break
                        
                        elif 'entities' in entry:
                            segment_time = entry.get('segment_time', 'Unknown')
                            entities_found = entry.get('entities_found', 0)
                            text_preview = entry.get('text_preview', '')
                            
                            if entities_found > 0:
                                print(f"[{segment_time}] Found {entities_found} entities")
                                print(f"   Text: {text_preview}")
                                
                                # Count entities by type
                                for entity in entry.get('entities', []):
                                    entity_type = entity.get('label', 'Unknown')
                                    entity_text = entity.get('text', '')
                                    entity_counts[f"{entity_type}: {entity_text}"] += 1
                                    total_entities += 1
                                
                                # Show some entities
                                shown_entities = entry.get('entities', [])[:3]  # Show first 3
                                for entity in shown_entities:
                                    print(f"     • {entity.get('text')} ({entity.get('label')})")
                                
                                if len(entry.get('entities', [])) > 3:
                                    print(f"     ... and {len(entry.get('entities', [])) - 3} more")
                                print()
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Error parsing JSON: {e}")
                        continue
                
                elif follow:
                    # No new line, wait a bit and try again
                    time.sleep(0.5)
                else:
                    # Not following and no more lines
                    break
    
    except KeyboardInterrupt:
        print(f"\n\n📊 Monitoring stopped. Total entities seen: {total_entities}")
        if entity_counts:
            print("🏆 Most frequent entities:")
            for entity, count in entity_counts.most_common(10):
                print(f"   • {entity} ({count})")
    
    except Exception as e:
        print(f"❌ Error monitoring file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Look for most recent real-time file
        transcripts_dir = Path("transcripts")
        if transcripts_dir.exists():
            realtime_files = list(transcripts_dir.glob("entities_realtime_*.jsonl"))
            if realtime_files:
                # Get the most recent file
                latest_file = max(realtime_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Found recent real-time file: {latest_file.name}")
                monitor_entities(str(latest_file))
            else:
                print("❌ No real-time entity files found in transcripts/")
                print("   Run a transcription first, then monitor with:")
                print("   python monitor_entities.py <path_to_realtime_file>")
        else:
            print("❌ No transcripts directory found")
    else:
        filepath = sys.argv[1]
        follow = len(sys.argv) < 3 or sys.argv[2].lower() != "nofollow"
        monitor_entities(filepath, follow)