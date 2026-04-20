# Ubuntu + NVIDIA Setup

This project can be copied from Windows to Ubuntu, but the Python environment must be rebuilt on Linux.

## What to copy from the Windows project

Copy these folders/files into the Linux machine:

- `checkpoints/`
- `configs/`
- `sam_audio_models/`
- `input/`
- optionally `output/`
- optionally `sam2/` if you want the local SAM2 source checkout there too
- optionally `sam-audio/` if you want the local SAM-Audio source checkout there too
- all project `.py` files and docs
- `tools/ffmpeg/linux/ffmpeg` if you want a repo-local Linux ffmpeg binary

Do **not** copy `.venv/` from Windows.

## Enter the container

If your lab machine provides NVIDIA PyTorch Singularity images, enter the container first:

```bash
singularity run --nv /path/to/pytorch:24.07-py3.sif
```

Once the prompt changes to `Singularity>`, you are already inside the container in that same terminal.

## System packages

Install Ubuntu packages first:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg build-essential git pkg-config libgl1 libglib2.0-0
```

Notes:
- `ffmpeg` is the default Linux choice.
- If you prefer a repo-local binary, place it at `tools/ffmpeg/linux/ffmpeg`.
- `libgl1` and `libglib2.0-0` help OpenCV imports work on Ubuntu.

## Create the Linux environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-webui.txt
```

## Install PyTorch for your CUDA version

Install Linux CUDA wheels that match the container you launched. Example shape:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If the container line is different, use the matching PyTorch index instead.

Install `torchcodec` from that same PyTorch wheel index before installing `sam-audio`:

```bash
pip install torchcodec==0.11.* --index-url https://download.pytorch.org/whl/cu124
```

Do not rely on a bare `pip install torchcodec` on Linux after a CUDA-specific PyTorch install. That can pull a
different TorchCodec CUDA wheel and lead to import errors like `Could not load libtorchcodec` or `libnvrtc.so.13`.

Container examples:
- `pytorch:24.07-py3.sif` ships CUDA `12.5.1`, so `cu124` is a reasonable PyTorch wheel target.
- `pytorch:25.08-py3.sif` and `pytorch:25.10-py3.sif` ship CUDA `13.0.0`. Use those only if you intentionally want a CUDA 13 stack.

## Install SAM2

Expected local source location if you keep the repo inside this project copy:

- `sam2/`

If you copied the `sam2` source checkout to Linux:

```bash
cd sam2
python -m pip install -e .
cd ..
```

If not, clone it first on Linux, then install it.

## Install SAM-Audio

Expected local source location if you keep the repo inside this project copy:

- `sam-audio/`

If you copied the `sam-audio` source checkout to Linux:

```bash
cd sam-audio
python -m pip install -e .
cd ..
```

If not, clone it first on Linux, then install it.

If the editable install overwrites your working `torchcodec` package, reinstall `torchcodec` from the same PyTorch
wheel index again afterward.

## Hugging Face login

If you want online fallback model access:

```bash
huggingface-cli login
```

If you are using only local model folders in `sam_audio_models/`, this is optional unless the selected model config still references remote helper assets.

## Local model and checkpoint layout

Expected portable repo layout:

- `checkpoints/sam2.1_hiera_tiny.pt`
- `configs/sam2.1/sam2.1_hiera_t.yaml`
- `sam_audio_models/<model_name>/...`

Examples:

- `sam_audio_models/small-tv`
- `sam_audio_models/base-tv`

The UI dropdown and CLI model-id override are designed to work with relative paths like `sam_audio_models/small-tv`, so keep that folder structure unchanged when copying the repo to Linux.

## ffmpeg behavior in this repo

The project now checks for ffmpeg in this order:

1. explicit CLI/UI override path
2. OS-matched repo-local binary
3. the other repo-local binary as a fallback
4. system `ffmpeg` on `PATH`

Repo-local paths:
- Windows prefers `tools/ffmpeg/windows/ffmpeg.exe`
- Ubuntu/Linux prefers `tools/ffmpeg/linux/ffmpeg`

## Quick validation

Run these checks after setup:

```bash
python -c "import cv2, numpy, scipy, pandas, PIL, matplotlib; print('base deps ok')"
python -c "import torch, torchvision, torchaudio; print('torch ok')"
python -c "import torchcodec; print('torchcodec ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import sam_audio; print('sam_audio ok')"
python -c "from path_layout import resolve_ffmpeg_binary; print(resolve_ffmpeg_binary())"
```

## First run suggestions

CLI:

```bash
python dataset_prep.py input/test1_TomScott.mp4 --output-dir output/prepared_test --ffmpeg-bin ffmpeg
```

UI:

```bash
streamlit run webui.py
```

## Selection guidance on Linux

Preferred options:
- Streamlit in-browser selection for day-to-day use
- selection JSON for CLI runs and reproducible experiments

Recommended Linux workflow:
1. Use `streamlit run webui.py` and select the target in the browser.
2. For CLI-only or batch runs, reuse a saved selection JSON instead of depending on popup windows.

CLI notes:
- `tracker.py` supports `--mode point` and `--mode bbox`
- `tracker.py` also supports `--selection-json ...` for non-interactive runs
- `--skip-mask-confirmation` is useful for scripted Linux runs once the prompt format is already verified

OpenCV popup selection may work on Ubuntu desktop systems, but it should be treated as optional desktop-only behavior, not the primary migration path.
