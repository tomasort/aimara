#!/usr/bin/env python3
"""Transcribe audio with optimized buffered entity extraction using OpenAI Whisper and spaCy."""

import whisper
import torch
import spacy
from pathlib import Path
import json
import sys
from datetime import datetime
import re
from collections import defaultdict, Counter
import threading
import queue
import time

def get_device():
    """Detect available device for processing."""
    print("\n🔍 Detecting available device...\n")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Metal Performance Shaders (MPS) detected - Using Apple GPU")
        print("   (M1/M2/M3 Mac detected)")
        print("   Note: Setting MPS fallback to avoid NaN issues")
        # Set MPS fallback to handle potential NaN issues
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
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

def load_spacy_model():
    """Load Spanish spaCy model for entity recognition."""
    try:
        # Try to load Spanish model
        nlp = spacy.load("es_core_news_sm")
        print("✅ Loaded Spanish spaCy model (es_core_news_sm)")
        return nlp
    except OSError:
        try:
            # Fallback to English model
            nlp = spacy.load("en_core_web_sm")
            print("⚠️  Spanish model not found, using English model (en_core_web_sm)")
            return nlp
        except OSError:
            print("❌ No spaCy models found. Please install with:")
            print("   python -m spacy download es_core_news_sm")
            print("   python -m spacy download en_core_web_sm")
            return None

class BufferedEntityExtractor:
    """Optimized entity extraction with buffering and batch processing."""
    
    def __init__(self, output_dir="transcripts", buffer_size=50, video_info=None):
        self.nlp = load_spacy_model()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.buffer_size = buffer_size
        
        # Video information for tracking
        self.video_info = video_info or {}
        
        # Buffer for segments waiting to be processed
        self.segment_buffer = []
        self.processed_segments = []
        
        # Entity storage with video tracking
        self.entities = defaultdict(lambda: defaultdict(list))
        self.entity_counts = defaultdict(Counter)
        self.entity_timeline = []
        self.entity_database = defaultdict(list)  # Complete entity database across videos
        
        # Initialize entity files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entities_file = self.output_dir / f"entities_buffered_{timestamp}.json"
        self.realtime_file = self.output_dir / f"entities_realtime_{timestamp}.jsonl"
        self.database_file = self.output_dir / f"entity_database_{timestamp}.json"
        
        # Create real-time file and write header
        with open(self.realtime_file, 'w', encoding='utf-8') as f:
            header = {
                "timestamp": datetime.now().isoformat(), 
                "event": "extraction_started", 
                "buffer_size": buffer_size,
                "video_info": self.video_info
            }
            f.write(json.dumps(header, ensure_ascii=False) + '\n')
        
        # Custom patterns for political/media content
        self.custom_patterns = {
            'POLITICAL_TITLES': re.compile(r'\b(presidente|ministro|gobernador|alcalde|diputado|senador|comandante)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            'ORGANIZATIONS': re.compile(r'\b(PSUV|MUD|AN|TSJ|FANB|GNB|SEBIN|CICPC)\b', re.IGNORECASE),
            'PROGRAMS': re.compile(r'\bprograma\s+(\d+)', re.IGNORECASE),
            'COUNTRIES': re.compile(r'\b(Venezuela|Colombia|Brasil|Argentina|Chile|Ecuador|Perú|Bolivia|Uruguay|Paraguay|Estados Unidos|Cuba)\b', re.IGNORECASE),
        }
        
        self.stats = {
            'segments_processed': 0,
            'batches_processed': 0,
            'total_entities': 0,
            'processing_time': 0
        }
    
    def add_segment(self, text, start_time, end_time):
        """Add segment to buffer for batch processing."""
        self.segment_buffer.append({
            'text': text,
            'start_time': start_time,
            'end_time': end_time
        })
        
        # Process buffer when it reaches buffer_size
        if len(self.segment_buffer) >= self.buffer_size:
            self.process_buffer()
    
    def process_buffer(self):
        """Process all segments in buffer at once (batch processing)."""
        if not self.segment_buffer or not self.nlp:
            return
        
        start_time = time.time()
        batch_texts = [segment['text'] for segment in self.segment_buffer]
        
        # Batch process with spaCy (much faster than individual processing)
        docs = list(self.nlp.pipe(batch_texts, batch_size=25, disable=['parser', 'tagger']))
        
        # Process each document with its corresponding segment info
        batch_entities = []
        
        for doc, segment in zip(docs, self.segment_buffer):
            segment_entities = {
                'timestamp': {'start': segment['start_time'], 'end': segment['end_time']},
                'text': segment['text'],
                'entities': []
            }
            
            # Extract spaCy entities
            for ent in doc.ents:
                # Filter out very common/short entities to reduce noise
                if len(ent.text.strip()) > 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'MISC']:
                    entity_data = {
                        'text': ent.text.strip(),
                        'label': ent.label_,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                    }
                    
                    segment_entities['entities'].append(entity_data)
                    
                    # Store with video tracking information
                    entity_occurrence = {
                        'timestamp': segment['start_time'],
                        'end_timestamp': segment['end_time'],
                        'segment_time': f"{segment['start_time']:.1f}s - {segment['end_time']:.1f}s",
                        'video_info': self.video_info,
                        'context': segment['text'],
                        'char_position': {'start': ent.start_char, 'end': ent.end_char}
                    }
                    
                    self.entities[ent.label_][ent.text.strip()].append(entity_occurrence)
                    self.entity_counts[ent.label_][ent.text.strip()] += 1
                    
                    # Add to global entity database
                    self.entity_database[ent.text.strip()].append({
                        'type': ent.label_,
                        'video_id': self.video_info.get('id', 'unknown'),
                        'video_title': self.video_info.get('title', 'unknown'),
                        'video_url': self.video_info.get('url', 'unknown'),
                        'timestamp_start': segment['start_time'],
                        'timestamp_end': segment['end_time'],
                        'context': segment['text'],
                        'extraction_time': datetime.now().isoformat()
                    })
            
            # Extract custom patterns (batch process these too)
            for pattern_name, pattern in self.custom_patterns.items():
                matches = pattern.finditer(segment['text'])
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 2:  # Filter short matches
                        entity_data = {
                            'text': entity_text,
                            'label': pattern_name,
                            'start_char': match.start(),
                            'end_char': match.end(),
                        }
                        
                        segment_entities['entities'].append(entity_data)
                        
                        # Store with video tracking information
                        entity_occurrence = {
                            'timestamp': segment['start_time'],
                            'end_timestamp': segment['end_time'],
                            'segment_time': f"{segment['start_time']:.1f}s - {segment['end_time']:.1f}s",
                            'video_info': self.video_info,
                            'context': segment['text'],
                            'char_position': {'start': match.start(), 'end': match.end()}
                        }
                        
                        self.entities[pattern_name][entity_text].append(entity_occurrence)
                        self.entity_counts[pattern_name][entity_text] += 1
                        
                        # Add to global entity database
                        self.entity_database[entity_text].append({
                            'type': pattern_name,
                            'video_id': self.video_info.get('id', 'unknown'),
                            'video_title': self.video_info.get('title', 'unknown'),
                            'video_url': self.video_info.get('url', 'unknown'),
                            'timestamp_start': segment['start_time'],
                            'timestamp_end': segment['end_time'],
                            'context': segment['text'],
                            'extraction_time': datetime.now().isoformat()
                        })
            
            if segment_entities['entities']:
                batch_entities.append(segment_entities)
        
        # Add to timeline
        self.entity_timeline.extend(batch_entities)
        self.processed_segments.extend(self.segment_buffer)
        
        # Write entities to real-time file immediately
        self.write_realtime_entities(batch_entities)
        
        # Update stats
        self.stats['segments_processed'] += len(self.segment_buffer)
        self.stats['batches_processed'] += 1
        self.stats['total_entities'] += sum(len(seg['entities']) for seg in batch_entities)
        self.stats['processing_time'] += time.time() - start_time
        
        # Clear buffer
        self.segment_buffer = []
        
        # Save periodically (every 5 batches = every ~250 segments)
        if self.stats['batches_processed'] % 5 == 0:
            self.save_entities()
            self.save_entity_database()
            self.print_progress()
    
    def write_realtime_entities(self, batch_entities):
        """Write entities to real-time JSONL file as they're discovered."""
        try:
            with open(self.realtime_file, 'a', encoding='utf-8') as f:
                for segment in batch_entities:
                    if segment['entities']:
                        # Write each entity as a separate line for easy streaming
                        realtime_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'batch': self.stats['batches_processed'],
                            'segment_time': f"{segment['timestamp']['start']:.1f}s",
                            'video_info': self.video_info,
                            'text_preview': segment['text'][:100] + "..." if len(segment['text']) > 100 else segment['text'],
                            'entities_found': len(segment['entities']),
                            'entities': segment['entities']
                        }
                        f.write(json.dumps(realtime_entry, ensure_ascii=False) + '\n')
                        f.flush()  # Ensure immediate write to disk
        except Exception as e:
            print(f"⚠️  Error writing real-time entities: {e}")
    
    def save_entity_database(self):
        """Save the comprehensive entity database with video tracking."""
        try:
            database_summary = {
                'last_updated': datetime.now().isoformat(),
                'video_info': self.video_info,
                'stats': {
                    'total_unique_entities': len(self.entity_database),
                    'total_occurrences': sum(len(occurrences) for occurrences in self.entity_database.values()),
                    'processing_stats': self.stats
                },
                'entities': dict(self.entity_database)
            }
            
            with open(self.database_file, 'w', encoding='utf-8') as f:
                json.dump(database_summary, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️  Error saving entity database: {e}")
    
    def finalize(self):
        """Process any remaining segments in buffer and save final results."""
        if self.segment_buffer:
            self.process_buffer()
        
        # Write completion marker to real-time file
        try:
            with open(self.realtime_file, 'a', encoding='utf-8') as f:
                completion_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'event': 'extraction_completed',
                    'final_stats': self.stats,
                    'top_entities': self.get_top_entities(3)
                }
                f.write(json.dumps(completion_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Error writing completion marker: {e}")
        
        self.save_entities()
        self.save_entity_database()
        self.print_final_stats()
    
    def save_entities(self):
        """Save current entities to JSON file."""
        entities_summary = {
            'extraction_time': datetime.now().isoformat(),
            'stats': self.stats,
            'entity_counts': dict(self.entity_counts),
            'top_entities': self.get_top_entities(),
            'recent_timeline': self.entity_timeline[-100:],  # Keep last 100 segments
        }
        
        # Save detailed entities separately to avoid huge files
        detailed_file = self.entities_file.with_suffix('.detailed.json')
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'entities_by_type': dict(self.entities),
                'full_timeline': self.entity_timeline,
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary
        with open(self.entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_summary, f, indent=2, ensure_ascii=False)
    
    def get_top_entities(self, top_n=10):
        """Get most frequently mentioned entities by category."""
        top_entities = {}
        for category, counter in self.entity_counts.items():
            top_entities[category] = counter.most_common(top_n)
        return top_entities
    
    def print_progress(self):
        """Print processing progress."""
        avg_time = self.stats['processing_time'] / max(self.stats['batches_processed'], 1)
        print(f"🔍 Entity extraction progress:")
        print(f"   Segments processed: {self.stats['segments_processed']}")
        print(f"   Batches processed: {self.stats['batches_processed']}")
        print(f"   Entities found: {self.stats['total_entities']}")
        print(f"   Avg batch time: {avg_time:.2f}s")
        print(f"   Real-time file: {self.realtime_file.name}")
        
        # Show recent top entities
        if self.entity_counts:
            print(f"   Recent top entities:")
            for category, counter in list(self.entity_counts.items())[:3]:
                top_3 = counter.most_common(3)
                if top_3:
                    entities_str = ", ".join([f"{name}({count})" for name, count in top_3])
                    print(f"     {category}: {entities_str}")
    
    def print_final_stats(self):
        """Print final extraction statistics."""
        print(f"\n🎯 Final Entity Extraction Results:")
        print(f"   Total segments: {self.stats['segments_processed']}")
        print(f"   Total entities: {self.stats['total_entities']}")
        print(f"   Processing time: {self.stats['processing_time']:.2f}s")
        print(f"   Entities per second: {self.stats['total_entities'] / max(self.stats['processing_time'], 1):.1f}")
        
        top_entities = self.get_top_entities(5)
        for category, entities in top_entities.items():
            if entities:
                print(f"\n   Top {category}:")
                for name, count in entities:
                    print(f"     • {name}: {count} mentions")

def extract_video_info_from_filename(audio_file):
    """Extract video information from the audio filename."""
    filename = Path(audio_file).stem
    
    # Try to extract program number and other info from filename
    video_info = {
        'filename': filename,
        'source_audio': str(audio_file)
    }
    
    # Extract program number if present
    import re
    program_match = re.search(r'Programa (\d+)', filename)
    if program_match:
        video_info['program_number'] = program_match.group(1)
    
    # Extract video ID from URL if available (would need to be passed separately)
    # For now, use a hash of the filename as a unique identifier
    import hashlib
    video_info['id'] = hashlib.md5(filename.encode()).hexdigest()[:8]
    video_info['title'] = filename
    
    # Try to extract date/time info if present
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if date_match:
        video_info['date'] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    
    return video_info

def transcribe_with_buffered_extraction(audio_file, model_name="small", device=None, language="es", buffer_size=50):
    """
    Transcribe audio file using Whisper with optimized buffered entity extraction.
    
    Args:
        audio_file: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda, mps) - auto-detected if None
        language: Language code (es for Spanish, en for English, etc.)
        buffer_size: Number of segments to buffer before batch processing
        
    Returns:
        Transcription result dictionary with entities
    """
    
    audio_path = Path(audio_file)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    # Initialize buffered entity extractor
    extractor = BufferedEntityExtractor(buffer_size=buffer_size)
    
    print(f"\n🎙️  Loading Whisper model: {model_name}")
    print(f"   Device: {device.upper()}")
    print(f"   Language: {language}")
    print(f"   Entity extraction: {'✅ Enabled' if extractor.nlp else '❌ Disabled (spaCy not available)'}")
    print(f"   Buffer size: {buffer_size} segments")
    
    try:
        # Load model with specified device
        model = whisper.load_model(model_name, device=device)
        
        print(f"\n📝 Transcribing audio file with buffered entity extraction...")
        print(f"   File: {audio_path.name}")
        print(f"   Size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"   Entities will be saved to: {extractor.entities_file}")
        print(f"\n   Processing with {buffer_size}-segment batches for optimal performance...\n")
        
        try:
            # Transcribe normally first (faster without custom callbacks)
            result = model.transcribe(
                str(audio_path),
                language=language,
                verbose=True,  # Show progress
                fp16=False,  # Disable FP16 for MPS compatibility
            )
            
            # Then process segments in batches for entity extraction
            if extractor.nlp and 'segments' in result:
                print(f"\n🔍 Processing {len(result['segments'])} segments for entity extraction...")
                print(f"   Using batch processing with {buffer_size}-segment buffers...")
                
                for i, segment in enumerate(result['segments']):
                    extractor.add_segment(
                        segment['text'],
                        segment['start'],
                        segment['end']
                    )
                    
                    # Show progress every 100 segments
                    if (i + 1) % 100 == 0:
                        print(f"   Buffered {i + 1}/{len(result['segments'])} segments...")
                
                # Process any remaining segments
                extractor.finalize()
            
            # Add entity data to result
            result['entities'] = {
                'entities_file': str(extractor.entities_file),
                'detailed_file': str(extractor.entities_file.with_suffix('.detailed.json')),
                'realtime_file': str(extractor.realtime_file),
                'stats': extractor.stats,
                'top_entities': extractor.get_top_entities(5)
            }
            
            return result
            
        except Exception as transcribe_error:
            if device == 'mps' and 'nan' in str(transcribe_error).lower():
                print(f"⚠️  MPS NaN error detected. Falling back to CPU...")
                print(f"   Original error: {transcribe_error}")
                
                # Reload model on CPU
                model = whisper.load_model(model_name, device='cpu')
                result = model.transcribe(
                    str(audio_path),
                    language=language,
                    verbose=True,
                    fp16=False,
                )
                
                # Process segments for entity extraction
                if extractor.nlp and 'segments' in result:
                    for segment in result['segments']:
                        extractor.add_segment(
                            segment['text'],
                            segment['start'],
                            segment['end']
                        )
                    extractor.finalize()
                
                result['entities'] = {
                    'entities_file': str(extractor.entities_file),
                    'detailed_file': str(extractor.entities_file.with_suffix('.detailed.json')),
                    'realtime_file': str(extractor.realtime_file),
                    'stats': extractor.stats,
                    'top_entities': extractor.get_top_entities(5)
                }
                
                return result
            else:
                raise transcribe_error
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

def save_transcript_with_entities(result, output_dir="transcripts"):
    """Save transcription results with entity information in multiple formats."""
    
    if result is None:
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Use timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"transcript_buffered_{timestamp}"
    
    # Save as plain text
    text_file = output_path / f"{base_name}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    print(f"\n✅ Plain text transcript saved:")
    print(f"   {text_file.resolve()}")
    
    # Save as JSON with segments, timestamps, and entity summary
    json_file = output_path / f"{base_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'text': result['text'],
            'language': result.get('language'),
            'segments': result.get('segments', []),
            'duration': result.get('duration'),
            'entities': result.get('entities', {}),
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Detailed transcript with entities saved:")
    print(f"   {json_file.resolve()}")
    
    # Save as SRT (subtitle format)
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
    
    # Print entity summary
    if 'entities' in result:
        entities = result['entities']
        stats = entities.get('stats', {})
        print(f"\n🔍 Entity Extraction Summary:")
        print(f"   Segments processed: {stats.get('segments_processed', 0)}")
        print(f"   Total entities found: {stats.get('total_entities', 0)}")
        print(f"   Processing time: {stats.get('processing_time', 0):.2f}s")
        print(f"   Entities file: {entities.get('entities_file', 'N/A')}")
        print(f"   Detailed file: {entities.get('detailed_file', 'N/A')}")
        print(f"   Real-time file: {entities.get('realtime_file', 'N/A')}")
        
        if 'top_entities' in entities:
            print(f"\n   Top entities by category:")
            for category, top_list in entities['top_entities'].items():
                if top_list:
                    print(f"     {category}:")
                    for entity, count in top_list[:3]:  # Show top 3
                        print(f"       • {entity} ({count} mentions)")
    
    return {
        'text_file': text_file,
        'json_file': json_file,
        'srt_file': srt_file,
        'entities_file': result.get('entities', {}).get('entities_file'),
        'detailed_entities_file': result.get('entities', {}).get('detailed_file'),
        'realtime_entities_file': result.get('entities', {}).get('realtime_file'),
    }

if __name__ == "__main__":
    # Default audio file
    audio_file = "audio/EN VIVO ｜ Con El Mazo Dando ｜ Diosdado Cabello ｜ Programa 548.mp3"
    
    # Allow custom audio file as argument
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    # Allow model selection as second argument
    model = sys.argv[2] if len(sys.argv) > 2 else "small"
    
    # Allow device override as third argument
    device = sys.argv[3].lower() if len(sys.argv) > 3 else None
    if device not in ['cpu', 'cuda', 'mps', None]:
        print(f"❌ Invalid device: {device}. Use 'cpu', 'cuda', 'mps', or leave empty for auto-detect")
        sys.exit(1)
    
    # Allow buffer size as fourth argument
    buffer_size = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    print("=" * 80)
    print("🎵 Whisper Audio Transcription with Optimized Buffered Entity Extraction")
    print("=" * 80)
    
    # Transcribe with buffered entity extraction
    result = transcribe_with_buffered_extraction(
        audio_file, 
        model_name=model, 
        device=device, 
        language="es",
        buffer_size=buffer_size
    )
    
    if result:
        # Save results
        files = save_transcript_with_entities(result)
        
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