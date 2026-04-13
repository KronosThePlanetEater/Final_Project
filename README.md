# Final_Project

Local pipeline for:
- SAM2-based visual tracking and mask export
- SAM-Audio visual prompting for audio separation
- evaluation and result analysis
- a local Streamlit UI for running experiments

## Project Layout
- `input/`: input videos and optional evaluation files
- `output/`: generated tracker, audio, eval, analysis, and UI run artifacts
- `checkpoints/`: SAM2 checkpoints
- `configs/`: SAM2 configs
- `sam_audio_models/`: local SAM-Audio model folders
- `tools/ffmpeg/`: optional repo-local ffmpeg binaries for Windows and Linux

## Before You Start
- Do **not** copy `.venv` between Windows and Linux.
- If you move the project to another machine, copy the project files and recreate `.venv` on that machine.
- Recommended local model layout:
  - `sam_audio_models/small-tv/checkpoint.pt`
  - `sam_audio_models/small-tv/config.json`
- Recommended SAM2 layout:
  - `checkpoints/sam2.1_hiera_tiny.pt`
  - `configs/sam2.1/sam2.1_hiera_t.yaml`

## Windows Setup
### 1. Open the project folder
Use PowerShell in:

```powershell
cd C:\Users\dhrum\Downloads\Final_Project
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install core project dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Install UI dependencies
```powershell
python -m pip install -r requirements-webui.txt
```

### 5. Install PyTorch
Install the PyTorch build that matches your CUDA setup. Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If you do not have a compatible NVIDIA/CUDA setup, install the CPU or correct CUDA build from PyTorch instead.

### 6. Install SAM2
If you already have a local `sam2` source folder, install it in editable mode:

```powershell
cd sam2
python -m pip install -e .
cd ..
```

If you do not have it yet, clone/download `sam2` first, then run the install command above.

### 7. Install SAM-Audio
If you already have a local `sam-audio` source folder, install it in editable mode:

```powershell
cd sam-audio
python -m pip install -e .
cd ..
```

If you do not have it yet, clone/download `sam-audio` first, then run the install command above.

### 8. Confirm ffmpeg
The project checks in this order:
1. explicit `--ffmpeg-bin` or UI override
2. `tools/ffmpeg/windows/ffmpeg.exe`
3. `tools/ffmpeg/linux/ffmpeg`
4. system `ffmpeg`

You can either:
- put `ffmpeg.exe` in `tools/ffmpeg/windows/ffmpeg.exe`, or
- install ffmpeg on Windows and add it to `PATH`

### 9. Optional Hugging Face login
Only needed if a run still tries to access online model assets:

```powershell
huggingface-cli login
```

### 10. Validate the environment
```powershell
python -c "import cv2, numpy, scipy, pandas, PIL, matplotlib; print('base deps ok')"
python -c "import torch, torchvision, torchaudio; print('torch ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import sam_audio; print('sam_audio ok')"
python -c "from path_layout import resolve_ffmpeg_binary; print(resolve_ffmpeg_binary())"
```

## Linux Setup (Ubuntu + NVIDIA)
### 1. Open the project folder
```bash
cd /path/to/Final_Project
```

### 2. Install system packages
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg build-essential git pkg-config libgl1 libglib2.0-0
```

### 3. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install core project dependencies
```bash
python -m pip install -r requirements.txt
```

### 5. Install UI dependencies
```bash
python -m pip install -r requirements-webui.txt
```

### 6. Install PyTorch
Install the PyTorch build that matches the Linux machine's CUDA version. Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If the Linux machine uses a different CUDA version, use the matching PyTorch wheel index instead.

### 7. Install SAM2
If the repo contains `sam2/`:

```bash
cd sam2
python -m pip install -e .
cd ..
```

Otherwise clone/download it first, then run the same install command.

### 8. Install SAM-Audio
If the repo contains `sam-audio/`:

```bash
cd sam-audio
python -m pip install -e .
cd ..
```

Otherwise clone/download it first, then run the same install command.

### 9. Confirm ffmpeg
The project checks in this order:
1. explicit `--ffmpeg-bin` or UI override
2. `tools/ffmpeg/linux/ffmpeg`
3. `tools/ffmpeg/windows/ffmpeg.exe`
4. system `ffmpeg`

Recommended Linux options:
- use `tools/ffmpeg/linux/ffmpeg`, or
- install system `ffmpeg` with `apt`

### 10. Optional Hugging Face login
```bash
huggingface-cli login
```

### 11. Validate the environment
```bash
python -c "import cv2, numpy, scipy, pandas, PIL, matplotlib; print('base deps ok')"
python -c "import torch, torchvision, torchaudio; print('torch ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import sam_audio; print('sam_audio ok')"
python -c "from path_layout import resolve_ffmpeg_binary; print(resolve_ffmpeg_binary())"
```

## What To Copy To Another Machine
Copy these if you want to avoid re-downloading large assets:
- `checkpoints/`
- `configs/`
- `sam_audio_models/`
- `input/`
- `tools/ffmpeg/`
- optionally `output/`
- optionally `sam2/`
- optionally `sam-audio/`

Do **not** copy:
- `.venv/`

## Running The Web UI
From an activated virtual environment:

```powershell
streamlit run webui.py
```

Open the local URL that Streamlit prints in the terminal.

## Running The Pipeline From Terminal
### 1. Tracker only
Interactive point selection example:

```powershell
python tracker.py input\test1_TomScott.mp4 output\terminal_runs\tracker_test1\video\tracked.mp4 --checkpoint checkpoints\sam2.1_hiera_tiny.pt --config configs\sam2.1\sam2.1_hiera_t.yaml --frames-dir output\terminal_runs\tracker_test1\frames --artifacts-dir output\terminal_runs\tracker_test1\artifacts --mode point --preview
```

### 2. Audio only
Use tracker masks plus the original video:

```powershell
python audio_pipeline.py --video-path input\test1_TomScott.mp4 --mask-path output\terminal_runs\tracker_test1\artifacts\masks\masks.npz --output-dir output\terminal_runs\audio_test1 --clip-id test1_TomScott --model-id sam_audio_models\small-tv --predict-spans --reranking-candidates 1
```

### 3. Evaluation
If you have clean reference audio and/or ground-truth masks:

```powershell
python evaluation.py --output-dir output\eval_test1 --clip-id test1_TomScott --model-id sam_audio_models\small-tv --predicted-mask-path output\terminal_runs\tracker_test1\artifacts\masks\masks.npz --estimated-audio-path output\terminal_runs\audio_test1\target.wav --reference-audio-path input\reference.wav --ground-truth-mask-path input\gt_masks.npz --audio-metadata-path output\terminal_runs\audio_test1\audio_run_metadata.json
```

### 4. Analysis
```powershell
python analyze_results.py output\ui_runs\SOME_JOB_ID\runs --output-dir output\analysis_test1
```

## Evaluation Inputs
- `Reference audio`: used for SI-SDR
- `Ground-truth masks`: used for IoU

If those are missing, evaluation still runs, but the corresponding metrics are skipped.

## Notes
- Use the Streamlit UI as the preferred cross-platform workflow.
- On Linux, browser-based selection is the recommended path.
- OpenCV popup selection is optional and may depend on desktop GUI support.
- Comparison experiments are better than single runs for meaningful analysis plots.
