# SAM-Audio Setup

This project now expects the official `sam_audio` Python package for real visual-prompted audio separation.

## Required pieces

- Python environment with `torch`, `torchaudio`, `opencv-python`, `numpy`, and `ffmpeg`
- Official SAM-Audio code installed so `from sam_audio import SAMAudio, SAMAudioProcessor` works
- Hugging Face authentication with access to the SAM-Audio model repo you want to load

## Recommended model ids

- `facebook/sam-audio-small-tv`
- `facebook/sam-audio-base-tv`
- `facebook/sam-audio-large-tv`

The `-tv` variants are the recommended defaults here because this project uses visual prompting from tracker masks.

## Expected workflow

1. Prepare a canonical video clip with `dataset_prep.py`
2. Track the visual target with `tracker.py`
3. Run `audio_pipeline.py --video-path ... --mask-path ...`
4. Evaluate with `evaluation.py`
5. Aggregate runs with `analyze_results.py`

## Important alignment rule

`audio_pipeline.py` expects the supplied `--video-path` to match the tracker mask stack exactly in:

- frame count
- frame resolution
- clip segment

If tracker ran on only part of a longer video, use the matching prepared video segment instead of the full source file.

## Placeholder mode

`--allow-placeholder` is smoke-test mode only. It extracts passthrough audio and does not perform source separation, so SI-SDR from placeholder runs should not be treated as a real metric.


## Local model folders

For portable local runs, keep SAM-Audio models under `sam_audio_models/` in the project root.

Recommended layout:

- `sam_audio_models/small-tv/`
- `sam_audio_models/base-tv/`
- `sam_audio_models/large-tv/`

When using a local model folder, the UI and CLI should point to relative model ids like:

- `sam_audio_models/small-tv`
- `sam_audio_models/base-tv`

This keeps the repo portable between Windows and Linux because the model path stays relative to the project root.

## Local source checkouts

If you want fully local installs instead of remote package resolution, keep source checkouts at:

- `sam2/`
- `sam-audio/`

Then install them in the active environment with `python -m pip install -e .` from each repo directory.
