#!/usr/bin/env python3
"""Transcribe audio with real-time entity extraction using OpenAI Whisper and spaCy."""

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

class EntityExtractor:
    """Real-time entity extraction from transcript segments."""
    
    def __init__(self, output_dir="transcripts"):
        self.nlp = load_spacy_model()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Entity storage
        self.entities = defaultdict(lambda: defaultdict(list))
        self.entity_counts = defaultdict(Counter)
        self.entity_timeline = []
        
        # Initialize entity file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entities_file = self.output_dir / f"entities_{timestamp}.json"
        
        # Custom patterns for political/media content
        self.custom_patterns = {
            'POLITICAL_TITLES': re.compile(r'\b(presidente|ministro|gobernador|alcalde|diputado|senador|comandante)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            'ORGANIZATIONS': re.compile(r'\b(PSUV|MUD|AN|TSJ|FANB|GNB|SEBIN|CICPC)\b', re.IGNORECASE),
            'PROGRAMS': re.compile(r'\bprograma\s+(\d+)', re.IGNORECASE),
            'COUNTRIES': re.compile(r'\b(Venezuela|Colombia|Brasil|Argentina|Chile|Ecuador|Perú|Bolivia|Uruguay|Paraguay|Estados Unidos|Cuba)\b', re.IGNORECASE),
        }
    
    def extract_entities_from_segment(self, text, start_time, end_time):
        """Extract entities from a single transcript segment."""
        if not self.nlp:
            return
        
        # Process with spaCy
        doc = self.nlp(text)
        
        segment_entities = {
            'timestamp': {'start': start_time, 'end': end_time},
            'text': text,
            'entities': []
        }
        
        # Extract spaCy entities
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'confidence': getattr(ent, 'score', 1.0)  # Some models provide confidence
            }
            
            segment_entities['entities'].append(entity_data)
            self.entities[ent.label_][ent.text].append({
                'timestamp': start_time,
                'context': text,
                'segment_time': f"{start_time:.1f}s - {end_time:.1f}s"
            })
            self.entity_counts[ent.label_][ent.text] += 1
        
        # Extract custom patterns
        for pattern_name, pattern in self.custom_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entity_text = match.group().strip()
                entity_data = {
                    'text': entity_text,
                    'label': pattern_name,
                    'description': f'Custom pattern: {pattern_name}',
                    'start_char': match.start(),
                    'end_char': match.end(),
                    'confidence': 0.9
                }
                
                segment_entities['entities'].append(entity_data)
                self.entities[pattern_name][entity_text].append({
                    'timestamp': start_time,
                    'context': text,
                    'segment_time': f"{start_time:.1f}s - {end_time:.1f}s"
                })
                self.entity_counts[pattern_name][entity_text] += 1
        
        # Add to timeline
        if segment_entities['entities']:
            self.entity_timeline.append(segment_entities)
            
            # Save entities in real-time (every few segments to avoid too many writes)
            if len(self.entity_timeline) % 10 == 0:
                self.save_entities()
                self.print_recent_entities()
    
    def save_entities(self):
        """Save current entities to JSON file."""
        entities_summary = {
            'extraction_time': datetime.now().isoformat(),
            'total_segments_processed': len(self.entity_timeline),
            'entity_counts': dict(self.entity_counts),
            'entities_by_type': dict(self.entities),
            'timeline': self.entity_timeline[-50:],  # Keep last 50 segments in main file
            'top_entities': self.get_top_entities()
        }
        
        with open(self.entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_summary, f, indent=2, ensure_ascii=False)
    
    def get_top_entities(self, top_n=10):
        """Get most frequently mentioned entities by category."""
        top_entities = {}
        for category, counter in self.entity_counts.items():
            top_entities[category] = counter.most_common(top_n)
        return top_entities
    
    def print_recent_entities(self):
        """Print recently extracted entities to console."""
        if self.entity_timeline:
            recent = self.entity_timeline[-5:]  # Last 5 segments
            print(f"\n🔍 Recent entities extracted (last 5 segments):")
            for segment in recent:
                if segment['entities']:
                    timestamp = segment['timestamp']
                    print(f"   [{timestamp['start']:.1f}s] Found {len(segment['entities'])} entities")
                    for entity in segment['entities'][:3]:  # Show first 3
                        print(f"      • {entity['text']} ({entity['label']})")

def custom_progress_callback(extractor):
    """Create a custom progress callback that extracts entities."""
    def progress_callback(segment):
        # Extract entities from this segment
        extractor.extract_entities_from_segment(
            segment['text'], 
            segment['start'], 
            segment['end']
        )
        
        # Print progress as usual
        start = segment['start']
        end = segment['end']
        text = segment['text']
        print(f"[{start:07.3f} --> {end:07.3f}] {text}")
        
    return progress_callback

def transcribe_with_entity_extraction(audio_file, model_name="small", device=None, language="es"):
    """
    Transcribe audio file using Whisper with real-time entity extraction.
    
    Args:
        audio_file: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda, mps) - auto-detected if None
        language: Language code (es for Spanish, en for English, etc.)
        
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
    
    # Initialize entity extractor
    extractor = EntityExtractor()
    
    print(f"\n🎙️  Loading Whisper model: {model_name}")
    print(f"   Device: {device.upper()}")
    print(f"   Language: {language}")
    print(f"   Entity extraction: {'✅ Enabled' if extractor.nlp else '❌ Disabled (spaCy not available)'}")
    
    try:
        # Load model with specified device
        model = whisper.load_model(model_name, device=device)
        
        print(f"\n📝 Transcribing audio file with real-time entity extraction...")
        print(f"   File: {audio_path.name}")
        print(f"   Size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"   Entities will be saved to: {extractor.entities_file}")
        print(f"\n   This may take several minutes, please wait...\n")
        
        # Custom transcription with entity extraction
        try:
            # First, load and preprocess audio
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            
            # Detect language if not specified
            if language is None:
                _, probs = model.detect_language(mel)
                language = max(probs, key=probs.get)
                print(f"Detected language: {language}")
            
            # Decode with custom callback for entity extraction
            options = whisper.DecodingOptions(
                language=language,
                fp16=False,  # Disable FP16 for MPS compatibility
            )
            
            # Full transcription
            result = model.transcribe(
                str(audio_path),
                language=language,
                verbose=False,  # We'll handle our own output
                fp16=False,
            )
            
            # Process each segment for entity extraction
            if extractor.nlp and 'segments' in result:
                print("🔍 Processing segments for entity extraction...")
                for segment in result['segments']:
                    extractor.extract_entities_from_segment(
                        segment['text'],
                        segment['start'],
                        segment['end']
                    )
            
            # Final save of all entities
            extractor.save_entities()
            
            # Add entity data to result
            result['entities'] = {
                'entities_file': str(extractor.entities_file),
                'total_entities': sum(sum(counter.values()) for counter in extractor.entity_counts.values()),
                'entity_types': list(extractor.entity_counts.keys()),
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
                        extractor.extract_entities_from_segment(
                            segment['text'],
                            segment['start'],
                            segment['end']
                        )
                
                extractor.save_entities()
                result['entities'] = {
                    'entities_file': str(extractor.entities_file),
                    'total_entities': sum(sum(counter.values()) for counter in extractor.entity_counts.values()),
                    'entity_types': list(extractor.entity_counts.keys()),
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
    base_name = f"transcript_with_entities_{timestamp}"
    
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
        print(f"\n🔍 Entity Extraction Summary:")
        print(f"   Total entities found: {entities.get('total_entities', 0)}")
        print(f"   Entity types: {', '.join(entities.get('entity_types', []))}")
        print(f"   Entities file: {entities.get('entities_file', 'N/A')}")
        
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
    
    print("=" * 80)
    print("🎵 Whisper Audio Transcription with Real-time Entity Extraction")
    print("=" * 80)
    
    # Transcribe with entity extraction
    result = transcribe_with_entity_extraction(audio_file, model_name=model, device=device, language="es")
    
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