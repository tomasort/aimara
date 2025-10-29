# Local Development Setup Guide

## Problem
Your Anaconda Python 3.8 is too old for the dependencies. You need Python 3.9 or newer.

## Solution: Use uv to Manage Python Version

`uv` can automatically download and manage Python versions for you. This is the easiest approach.

### Step 1: Remove Old Environment
```bash
cd /Users/tomasortega/Projects/aimara
rm -rf .venv
```

### Step 2: Let uv Create Everything
```bash
# uv will use pyproject.toml and create the right Python version
uv sync
```

This will:
- ✅ Download Python 3.11 (latest stable)
- ✅ Create a `.venv` 
- ✅ Install all dependencies from `pyproject.toml`

### Step 3: Run Scripts with uv
```bash
# Option A: Run scripts directly with uv (recommended)
uv run python scripts/fetch_playlist.py

# Option B: Activate the venv manually
source .venv/bin/activate
python scripts/fetch_playlist.py
deactivate
```

## Why This Works Better Than Anaconda

| Feature | Anaconda | uv |
|---------|----------|-----|
| Python Version | Stuck at 3.8 | Can use 3.11+ |
| Package Manager | conda/pip mix | Pure uv (faster) |
| Virtual Env | Complex | Automatic |
| Reproducibility | Difficult | `uv.lock` file |

## Troubleshooting

### "uv not found"
Install uv first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Still getting Python 3.8 errors
Make sure you're not using Anaconda's Python:
```bash
which python  # Should NOT show anaconda path
python --version  # Should be 3.11+
```

If it shows Anaconda, remove it from PATH or use:
```bash
uv run python scripts/fetch_playlist.py  # This forces uv's Python
```

## Next Steps

Once `uv sync` completes successfully:

1. Test fetch_playlist:
```bash
uv run python scripts/fetch_playlist.py
```

2. Test download_audio (with your playlist):
```bash
uv run python scripts/download_audio.py
```

3. Test transcribe_audio:
```bash
uv run python scripts/transcribe_audio.py
```

## File Structure

```
aimara/
├── pyproject.toml       # Project config + dependencies
├── uv.lock             # Lock file (auto-created by uv)
├── .venv/              # Virtual environment (auto-created)
├── .python-version     # Python version marker
├── scripts/
│   ├── fetch_playlist.py
│   ├── download_audio.py
│   └── transcribe_audio.py
└── ...
```

## Running on M1 Mac

Same setup works! Just run:
```bash
uv sync
uv run python scripts/transcribe_audio.py
```

It will auto-detect MPS and use GPU acceleration. ✨
