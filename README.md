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
- Do **not** copy a Linux Conda env such as `.conda/` between machines or containers.
- If you move the project to another machine, copy the project files and recreate the environment on that machine.
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

### 2. Start the container
If you are using the Rutgers Singularity images, enter the container first:

```bash
singularity run --nv /path/to/pytorch:24.07-py3.sif
```

After this, the prompt changes to `Singularity>` and that same terminal is now inside the container.

### 3. Create and activate a project-local Conda environment
Use a project-local prefix such as `.conda` instead of `python -m venv`. This avoids the missing `python3-venv`
problem inside some containers.

If `conda` is not installed in the container, install Miniconda or Mambaforge in your home directory first, then
return to these commands from a new `Singularity>` shell.

If `conda` is already available in the container or your shell:

```bash
conda env create --prefix ./.conda -f environment-linux.yml
conda activate /path/to/Final_Project/.conda
python -m pip install --upgrade pip
```

If `conda activate` is not available yet in the current shell, initialize it first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create --prefix ./.conda -f environment-linux.yml
conda activate /path/to/Final_Project/.conda
python -m pip install --upgrade pip
```

Keep one environment per container line when possible. For example, recreate `.conda` if you switch from
`pytorch:24.07-py3.sif` to `pytorch:25.10-py3.sif`.

### 4. Core and UI dependencies
`environment-linux.yml` already installs both `requirements.txt` and `requirements-webui.txt`.

This project pins `streamlit==1.18.0`, which also requires `altair<5`. `requirements-webui.txt` already includes that pin.

If you later update either requirements file, refresh the active environment with:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-webui.txt
```

### 5. Install PyTorch
Install the PyTorch build that matches the container you launched, not the host shell outside the container. Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If you use a different PyTorch container line, choose the matching wheel index instead.

Install `torchcodec` from that same index before installing `sam-audio`. On Linux, a bare `pip install torchcodec`
or a transitive install pulled in by `sam-audio` can grab a different CUDA wheel and fail later with errors like
`Could not load libtorchcodec` or `libnvrtc.so.13`.

```bash
pip install torchcodec==0.11.* --index-url https://download.pytorch.org/whl/cu124
```

Container examples:
- `pytorch:24.07-py3.sif` ships CUDA `12.5.1`, so `cu124` is usually the closest stable PyTorch wheel family.
- `pytorch:25.08-py3.sif` and `pytorch:25.10-py3.sif` ship CUDA `13.0.0`; use those only if you intentionally want a CUDA 13 stack.

### 6. Install SAM2
If the repo contains `sam2/`:

```bash
cd sam2
python -m pip install -e .
cd ..
```

Otherwise clone/download it first, then run the same install command.

### 7. Install SAM-Audio
If the repo contains `sam-audio/`:

```bash
cd sam-audio
python -m pip install -e .
cd ..
```

Otherwise clone/download it first, then run the same install command.

If `sam-audio` tries to replace your already-correct `torchcodec` install, reinstall `torchcodec` from the same
PyTorch index after the editable install.

### 8. Confirm ffmpeg
The project checks in this order:
1. explicit `--ffmpeg-bin` or UI override
2. `tools/ffmpeg/linux/ffmpeg`
3. `tools/ffmpeg/windows/ffmpeg.exe`
4. system `ffmpeg`

Recommended Linux options:
- use `tools/ffmpeg/linux/ffmpeg`, or
- install system `ffmpeg` with `apt`

### 9. Optional Hugging Face login
```bash
huggingface-cli login
```

### 10. Validate the environment
```bash
python -c "import cv2, numpy, scipy, pandas, PIL, matplotlib; print('base deps ok')"
python -c "import torch, torchvision, torchaudio; print('torch ok')"
python -c "import torchcodec; print('torchcodec ok')"
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
- `.conda/`

## Running The Web UI
From an activated environment:

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
