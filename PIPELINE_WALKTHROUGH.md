#!/usr/bin/env python3
"""
AIMARA PIPELINE - COMPLETE WALKTHROUGH
========================================

This document explains step-by-step how the entire pipeline works.

USER COMMAND:
    uv run aimara.py pipeline-process PLAYLIST_URL --max-videos 2 --chunk-length 600

EXECUTION FLOW:
"""

# ==============================================================================
# STAGE 0: COMMAND-LINE INTERFACE (aimara.py)
# ==============================================================================

"""
1. User runs: uv run aimara.py pipeline-process <URL> [OPTIONS]
   
   aimara.py is loaded:
   - Click CLI framework parses arguments
   - Validates playlist URL
   - Sets defaults: model=small, language=es, device=auto, buffer_size=50, chunk_length=600
   
   Example parsed arguments:
   {
       'playlist_url': 'https://www.youtube.com/playlist?list=...',
       'max_videos': 2,
       'model': 'small',
       'language': 'es',
       'device': 'auto',
       'buffer_size': 50,
       'chunk_length': 600
   }

2. CLI displays header:
   ================================================================================
   🚀 Aimara Pipeline - Producer-Consumer Architecture
   ================================================================================
   📺 Playlist: https://www.youtube.com/playlist?list=...
   📊 Max videos: 2
   🤖 Whisper model: small
   🌐 Language: es
   💻 Device: auto
   📦 Buffer size: 50
   ✂️  Chunk length: 600s (~10 minutes)
   ================================================================================

3. Imports pipeline.py and creates Pipeline instance:
   pipeline = Pipeline(output_dir='.')
   
   Pipeline.__init__() does:
   - Creates directories: audio/, transcripts/, video_transcripts/
   - Sets up paths for later use
"""

# ==============================================================================
# STAGE 1: PIPELINE INITIALIZATION (pipeline.py)
# ==============================================================================

"""
4. pipeline.run() is called with all parameters
   
   Creates three thread-safe queues:
   
   download_queue (maxsize=5)
   ├─ Holds AudioFile objects
   └─ Producer: DownloadProducer
   └─ Consumer: TranscriptionWorker
   
   transcription_queue (maxsize=50)
   ├─ Holds TranscriptChunk objects
   └─ Producer: TranscriptionWorker
   └─ Consumer: EntityConsumer

5. THREE THREADS ARE CREATED:

   Thread 1: DownloadProducer
   ├─ Fetches playlist
   ├─ Downloads videos one by one
   ├─ Splits audio into chunks (600s each)
   └─ Puts AudioFile objects into download_queue

   Thread 2: TranscriptionWorker
   ├─ Loads Whisper model (small, device=auto)
   ├─ Consumes AudioFile from download_queue
   ├─ Transcribes each chunk
   ├─ WRITES transcript to file immediately
   └─ Puts TranscriptChunk into transcription_queue

   Thread 3: EntityConsumer
   ├─ Loads spaCy model (es_core_news_sm)
   ├─ Consumes TranscriptChunk from transcription_queue
   ├─ Extracts entities in batches (buffer_size=50)
   ├─ Writes JSONL in real-time
   └─ Aggregates per-video files

6. All three threads are started with .start()
   Main thread waits for completion with .join()
"""

# ==============================================================================
# STAGE 2: PARALLEL EXECUTION - DOWNLOAD PRODUCER
# ==============================================================================

"""
Thread: DownloadProducer
Status: RUNNING

Step 1: Fetch Playlist
   - Uses yt_dlp to extract_info() from playlist URL
   - Downloads metadata only (not video/audio yet)
   - Gets list of entries (videos in playlist)
   - Limits to max_videos=2
   
   Example: [Video 1, Video 2]

Step 2: For each video:
   
   VIDEO 1: "Con El Mazo Dando - Programa 548"
   ├─ video_id: "xyz123"
   ├─ title: "Con El Mazo Dando ｜ Diosdado Cabello ｜ Programa 548"
   ├─ duration: 14400 (4 hours)
   
   Step 2a: Download audio (yt-dlp FFmpegExtractAudio)
   ├─ Downloads best audio stream
   ├─ Converts to MP3 (192 kbps)
   ├─ Saves to: audio/Video_Title.mp3
   
   Step 2b: Split audio into chunks
   ├─ Calls split_audio_file()
   ├─ Uses ffprobe to get duration (14400s)
   ├─ Splits into: 14400s / 600s = 24 chunks
   ├─ Creates:
   │  ├─ audio/Video_Title_chunk_000.mp3 (600s)
   │  ├─ audio/Video_Title_chunk_001.mp3 (600s)
   │  ├─ ...
   │  └─ audio/Video_Title_chunk_023.mp3 (600s)
   
   Step 2c: Queue chunks for transcription
   ├─ For each chunk file created:
   │  ├─ Create AudioFile object
   │  ├─ Put into download_queue
   │  ├─ Log: "Queued: AudioFile(Video_Title chunk 0)"
   │  └─ TranscriptionWorker immediately starts processing
   
   Example first chunk:
   AudioFile(
       path='audio/Video_Title_chunk_000.mp3',
       video_id='xyz123',
       title='Con El Mazo Dando | ...',
       duration=14400
   )

Step 3: Repeat for VIDEO 2
   (Same process for second video)

Step 4: Signal completion
   ├─ After all videos queued
   ├─ Put None into download_queue (sentinel value)
   ├─ Signals TranscriptionWorker: "No more files coming"
   └─ Thread exits

QUEUE STATE DURING DOWNLOAD:
   download_queue: [
       AudioFile(chunk_0), AudioFile(chunk_1), AudioFile(chunk_2),
       AudioFile(chunk_3), AudioFile(chunk_4), ... (maxsize=5)
   ]
"""

# ==============================================================================
# STAGE 3: PARALLEL EXECUTION - TRANSCRIPTION WORKER
# ==============================================================================

"""
Thread: TranscriptionWorker
Status: RUNNING (in parallel with DownloadProducer)

Step 1: Load Whisper model
   ├─ whisper.load_model('small', device='auto')
   ├─ Auto-detect device:
   │  ├─ Check if MPS available (Apple Silicon) -> use MPS
   │  ├─ Check if CUDA available (NVIDIA) -> use CUDA
   │  └─ Fallback to CPU
   ├─ Load small model (~500MB)
   ├─ Log: "Model loaded successfully"

Step 2: Consume AudioFile from queue (with timeout=1s)
   ├─ audio_file = download_queue.get(timeout=1)
   ├─ If timeout -> continue (check for sentinel)
   ├─ If None (sentinel) -> break loop and exit
   ├─ Otherwise -> process file

   FIRST CHUNK ARRIVES (while DownloadProducer still downloading VIDEO 2):
   
   AudioFile(
       path='audio/Video_Title_chunk_000.mp3',
       video_id='xyz123',
       title='Con El Mazo Dando | ...'
   )

Step 3: Transcribe chunk
   ├─ Call whisper_model.transcribe(
   │      path='audio/Video_Title_chunk_000.mp3',
   │      language='es',
   │      verbose=False,
   │      fp16=False
   │  )
   ├─ Whisper processes the 600 seconds of audio
   ├─ Returns result dict with segments
   
   Example result:
   {
       'text': 'Buenos días... (full text)',
       'language': 'es',
       'segments': [
           {'start': 0.0, 'end': 5.2, 'text': 'Buenos días'},
           {'start': 5.2, 'end': 12.1, 'text': 'Bienvenidos a...'},
           ...
       ]
   }

Step 4: **IMMEDIATELY WRITE TRANSCRIPT**
   ├─ Get transcript file for this video:
   │  └─ transcripts/Con_El_Mazo_Dando_transcript_building.txt
   ├─ Combine all segment texts: 'Buenos días\nBienvenidos a\n...'
   ├─ Append to file (not overwrite!)
   ├─ f.flush() -> ensures data written to disk
   ├─ Log: "✓ Wrote transcript chunk to: transcripts/Con_El_Mazo_Dando_..."
   
   FILE GROWS IN REAL TIME:
   Initial:
   ├─ transcripts/Con_El_Mazo_Dando_transcript_building.txt (empty)
   
   After chunk 0:
   ├─ transcripts/Con_El_Mazo_Dando_transcript_building.txt (600s of text)
   
   After chunk 1:
   ├─ transcripts/Con_El_Mazo_Dando_transcript_building.txt (1200s of text)

Step 5: Create TranscriptChunk object
   ├─ Wrap segments in TranscriptSegment objects
   ├─ Include reference to original AudioFile
   ├─ Store full concatenated text
   
   TranscriptChunk(
       audio_file=AudioFile(...),
       segments=[
           TranscriptSegment(text='Buenos días', start=0.0, end=5.2, ...),
           TranscriptSegment(text='Bienvenidos a...', start=5.2, end=12.1, ...),
           ...
       ],
       full_text='Buenos días\nBienvenidos a\n...'
   )

Step 6: Queue for entity extraction
   ├─ transcription_queue.put(transcript_chunk)
   ├─ EntityConsumer wakes up and starts processing
   ├─ Log: "Finished transcribing and queuing: AudioFile(...)"

Step 7: Loop back to Step 2
   ├─ Consume next AudioFile from download_queue
   ├─ **While EntityConsumer processes the PREVIOUS chunk!**

EFFICIENCY EXAMPLE:
   Time T=0:    Download Video 1 chunks
   Time T=600:  Transcribe chunk 0 + Write to file
   Time T=605:  Queue chunk 0 for entities
               Download chunk 1 (still downloading)
   Time T=610:  Extract entities from chunk 0
   Time T=1200: Transcribe chunk 1 + Write to file
   Time T=1205: Queue chunk 1 for entities
   Time T=1210: Extract entities from chunk 1
   
   NO WAITING! Everything runs in parallel.
"""

# ==============================================================================
# STAGE 4: PARALLEL EXECUTION - ENTITY CONSUMER
# ==============================================================================

"""
Thread: EntityConsumer
Status: RUNNING (in parallel with Download + Transcription)

Step 1: Load spaCy model
   ├─ spacy.load('es_core_news_sm')
   ├─ Spanish NER model
   ├─ Disable parser and tagger for speed
   ├─ Log: "Loaded spaCy model: es_core_news_sm"

Step 2: Consume TranscriptChunk from queue
   ├─ chunk = transcription_queue.get(timeout=1)
   ├─ FIRST CHUNK ARRIVES (from TranscriptionWorker)
   
   TranscriptChunk(
       audio_file=AudioFile(path='...chunk_000.mp3', video_id='xyz123', ...),
       segments=[...],
       full_text='...'
   )

Step 3: Process all segments in chunk
   ├─ For each segment in chunk.segments:
   │  ├─ Add to segment_buffer
   │  ├─ Increment segment_count
   │  ├─ Check if buffer full (buffer_size=50)
   │  └─ If full -> call _process_buffer()

   BUFFERING EXAMPLE:
   ├─ Segment 1 added (buffer has 1)
   ├─ Segment 2 added (buffer has 2)
   ├─ ...
   ├─ Segment 50 added (buffer has 50) -> PROCESS!
   │  ├─ Extract entities from all 50 segments
   │  ├─ Write to JSONL
   │  ├─ Update per-video tracking
   │  ├─ Clear buffer
   └─ Segment 51 added (buffer has 1)

Step 4: _process_buffer() - Batch entity extraction
   ├─ Get batch of 50 segments
   ├─ Extract text from each segment
   ├─ Batch process with spaCy:
   │  └─ docs = nlp.pipe(batch_texts, batch_size=25, disable=[...])
   │     (Faster than processing one-by-one)
   
   ├─ For each (doc, segment) pair:
   │  └─ Extract named entities:
   │     ├─ PERSON: "Diosdado Cabello"
   │     ├─ ORG: "PSUV"
   │     ├─ GPE: "Venezuela"
   │     └─ MISC: (other)
   
   │  └─ Extract custom patterns:
   │     ├─ POLITICAL_TITLES: "presidente Maduro"
   │     ├─ ORGANIZATIONS: "FANB", "GNB", "SEBIN"
   │     ├─ PROGRAMS: "programa 548"
   │     └─ COUNTRIES: geographic names
   
   ├─ For each entity found:
   │  ├─ Increment entity count
   │  ├─ Add to global entity_database
   │  ├─ Add to video-specific entity_database
   │  └─ Call _write_entity() to JSONL

Step 5: _write_entity() - Real-time JSONL output
   ├─ Open entities_YYYYMMDD_HHMMSS.jsonl in append mode
   ├─ Write entity as JSON line:
   │  {
   │      "entity": "Diosdado Cabello",
   │      "type": "PERSON",
   │      "video_info": {"title": "...", "video_id": "xyz123", ...},
   │      "timestamp_start": 120.5,
   │      "timestamp_end": 125.3,
   │      "context": "Diosdado Cabello habló...",
   │      "extraction_time": "2025-10-28T15:30:45.123456"
   │  }
   ├─ f.flush() -> disk write
   ├─ Result: File grows in real-time!

Step 6: Track per-video
   ├─ Store entities in: video_entities[video_id][entity_name]
   ├─ Store segments in: video_segments[video_id]
   ├─ Used later for per-video files

Step 7: Loop back
   ├─ Check if buffer >= 50 -> process if full
   ├─ Consume next chunk
   ├─ Continue until sentinel (None) received

REAL-TIME OBSERVATION:
   Terminal 1 (running pipeline):
   INFO [EntityConsumer] Processing buffer with 50 segments
   INFO [EntityConsumer] Found 234 entities in batch
   
   Terminal 2 (monitoring files):
   $ tail -f transcripts/entities_*.jsonl
   {"entity": "Diosdado", "type": "PERSON", ...}
   {"entity": "PSUV", "type": "ORG", ...}
   {"entity": "Venezuela", "type": "GPE", ...}
   (new entities appearing in real-time!)
"""

# ==============================================================================
# STAGE 5: FINALIZATION
# ==============================================================================

"""
When all three threads complete:

DownloadProducer: FINISHED
├─ All videos queued
└─ Sent sentinel (None) to download_queue

TranscriptionWorker: FINISHED
├─ Received sentinel
├─ Sent sentinel to transcription_queue
└─ Exited

EntityConsumer: FINALIZE
├─ Received sentinel
├─ Process remaining buffer segments
├─ Call _finalize()

_finalize() does:

1. Process remaining buffer
   ├─ If buffer has < 50 segments -> still process them!
   ├─ Don't lose data!

2. Save per-video transcript files
   ├─ For each video_id in video_segments:
   │  ├─ Read intermediate file: Con_El_Mazo_Dando_transcript_building.txt
   │  ├─ Create final files:
   │  │  ├─ video_transcripts/Con_El_Mazo_Dando_transcript.txt (full text)
   │  │  ├─ video_transcripts/Con_El_Mazo_Dando_transcript.json (segmented)
   │  │  └─ video_transcripts/Con_El_Mazo_Dando_entities.json (entities)
   │  └─ Delete intermediate _building.txt file

3. Save global summary
   ├─ transcripts/entities_summary_YYYYMMDD_HHMMSS.json
   ├─ Contains:
   │  ├─ total_segments_processed: 5000
   │  ├─ total_entities_found: 2340
   │  ├─ entity_counts by category
   │  └─ timestamp

4. Return results to CLI
   └─ {
         'entities_file': '.../entities_YYYYMMDD_HHMMSS.jsonl',
         'summary_file': '.../entities_summary_YYYYMMDD_HHMMSS.json',
         'video_transcripts_dir': '.../video_transcripts/',
         'total_entities': 2340,
         'total_segments': 5000,
         'total_videos': 2
      }
"""

# ==============================================================================
# STAGE 6: CLI OUTPUT
# ==============================================================================

"""
CLI displays final results:

✅ Pipeline completed successfully!

📊 Results:
   🎬 Total segments processed: 5000
   🎥 Total videos processed: 2
   🔍 Total entities extracted: 2340

📁 Output files:
   📄 Global entities (JSONL): ./transcripts/entities_20251028_153045.jsonl
   📊 Global summary: ./transcripts/entities_summary_20251028_153045.json
   📂 Per-video files: ./video_transcripts/
      ├─ VIDEO_transcript.txt (full transcript)
      ├─ VIDEO_transcript.json (segmented transcript)
      └─ VIDEO_entities.json (entities for that video)
"""

# ==============================================================================
# DIRECTORY STRUCTURE AFTER COMPLETION
# ==============================================================================

"""
./
├── audio/
│   ├── Video_Title_1.mp3
│   ├── Video_Title_1_chunk_000.mp3
│   ├── Video_Title_1_chunk_001.mp3
│   ├── ...
│   ├── Video_Title_2.mp3
│   ├── Video_Title_2_chunk_000.mp3
│   └── ...
├── transcripts/
│   ├── entities_20251028_153045.jsonl
│   │   ├─ Line 1: {"entity": "Diosdado", ...}
│   │   ├─ Line 2: {"entity": "PSUV", ...}
│   │   └─ ... (2340 lines)
│   └── entities_summary_20251028_153045.json
│       └─ {"total_entities": 2340, "entity_counts": {...}, ...}
└── video_transcripts/
    ├── Video_Title_1_transcript.txt
    │   └─ (full transcript of video 1)
    ├── Video_Title_1_transcript.json
    │   └─ {"segments": [...], "text": "...", ...}
    ├── Video_Title_1_entities.json
    │   └─ {"Diosdado": {...}, "PSUV": {...}, ...}
    ├── Video_Title_2_transcript.txt
    ├── Video_Title_2_transcript.json
    └── Video_Title_2_entities.json
"""

# ==============================================================================
# KEY PARALLELISM MOMENTS
# ==============================================================================

"""
Why this is SO much faster than sequential processing:

SEQUENTIAL (OLD):
   Download Video 1 (600s) -> Transcribe Video 1 (3000s) -> Extract entities (500s)
   Download Video 2 (600s) -> Transcribe Video 2 (3000s) -> Extract entities (500s)
   TOTAL: ~9000 seconds (2.5 hours)

PARALLEL (NEW):
   Time 0-600s:    Download Video 1 + Video 2 chunks
   Time 600-1200s: Transcribe chunk 1 + Download chunk 3 + Extract entities from chunk 0
   Time 1200-1800: Transcribe chunk 2 + Download chunk 4 + Extract entities from chunk 1
   ...
   TOTAL: ~2000-2500 seconds (30-40 minutes)
   
   SPEEDUP: 4-5x faster!
"""

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

"""
What happens if something fails:

1. MPS NaN Error (GPU problem):
   ├─ Caught in TranscriptionWorker
   ├─ Falls back to CPU
   ├─ Continues processing (slower but works)

2. Network Error (download fails):
   ├─ Caught in DownloadProducer
   ├─ Logged with error message
   ├─ Continues with next video

3. File I/O Error:
   ├─ Caught in EntityConsumer._write_entity()
   ├─ Logged but doesn't stop processing
   ├─ Critical files (final JSON) still get saved

4. spaCy model missing:
   ├─ Caught in EntityConsumer.__init__()
   ├─ Logs warning
   ├─ Only spaCy entities skipped, patterns still work

5. Sentinel handling:
   ├─ None in queue signals end
   ├─ Each thread checks for None
   ├─ Gracefully exits
"""

print("=" * 80)
print("WALKTHROUGH COMPLETE")
print("=" * 80)
print("""
The pipeline is a three-stage producer-consumer system:

1. DownloadProducer (Thread 1)
   ├─ Downloads videos from playlist
   ├─ Splits into chunks for faster processing
   └─ Queues AudioFile objects

2. TranscriptionWorker (Thread 2)
   ├─ Consumes AudioFile
   ├─ Transcribes with Whisper
   ├─ **WRITES TRANSCRIPT IMMEDIATELY**
   └─ Queues TranscriptChunk

3. EntityConsumer (Thread 3)
   ├─ Consumes TranscriptChunk
   ├─ Extracts entities in batches (buffer_size=50)
   ├─ **WRITES ENTITIES TO JSONL IN REAL-TIME**
   └─ Aggregates per-video files

MAXIMUM PARALLELISM:
- While downloading video 3, transcribing video 2, extracting entities from video 1
- All three stages running simultaneously
- Thread-safe queues handle communication
- Real-time file output allows monitoring

RUN COMMAND:
    uv run aimara.py pipeline-process PLAYLIST_URL --max-videos 2 --chunk-length 600
""")
