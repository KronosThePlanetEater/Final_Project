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
- `sam2/`: optional project-local SAM2 source checkout
- `sam-audio/`: optional project-local SAM-Audio source checkout
- `tools/ffmpeg/`: optional repo-local ffmpeg binaries for Windows and Linux

## External Artifacts
Large artifacts that are not convenient to keep directly in Git are available in this Rutgers Box folder:

- https://rutgers.box.com/s/i2tkqmj9vd429yup5v41tsmd1m01yqop

This folder is intended for large project assets such as SAM-Audio model folders and saved run outputs. Hugging Face authentication is still handled separately with `hf auth login`; do not commit Hugging Face tokens to this repository or include them in shared documentation. If this Box folder currently contains a token file, remove it and revoke that token before sharing the folder more broadly.

## Before You Start
- Do **not** copy `.venv` between Windows and Linux.
- If you move the project to another machine, copy the project files and recreate `.venv` on that machine.
- Use the same CUDA wheel family on Windows and Linux. This project standardizes on **CUDA 12.8 / PyTorch `cu128` wheels**.
- Avoid mixing `cu124`, `cu126`, `cu128`, and CUDA 13 wheels in the same `.venv`.
- Hugging Face authentication is required for real SAM-Audio model loading. Do not commit tokens into this repo.
- Use project-relative model paths when possible so the repo stays portable between Windows and Linux.
- Recommended local SAM-Audio model layout:
  - `sam_audio_models/small-tv/checkpoint.pt`
  - `sam_audio_models/small-tv/config.json`
  - `sam_audio_models/base-tv/checkpoint.pt`
  - `sam_audio_models/base-tv/config.json`
  - `sam_audio_models/large-tv/checkpoint.pt`
  - `sam_audio_models/large-tv/config.json`
- Recommended SAM2 layout:
  - `checkpoints/sam2.1_hiera_tiny.pt`
  - `configs/sam2.1/sam2.1_hiera_t.yaml`

## Expected Workflow
1. Prepare a canonical video clip with `dataset_prep.py` when needed.
2. Track the visual target with `tracker.py`.
3. Run `audio_pipeline.py --video-path ... --mask-path ...`.
4. Evaluate with `evaluation.py`.
5. Aggregate runs with `analyze_results.py`.

Important alignment rule: `audio_pipeline.py` expects the supplied `--video-path` to match the tracker mask stack exactly in frame count, frame resolution, and clip segment. If tracking ran on only part of a longer video, use the matching prepared video segment instead of the full source file.

Placeholder mode is smoke-test mode only. `--allow-placeholder` extracts passthrough audio and does not perform source separation, so SI-SDR from placeholder runs should not be treated as a real metric.

## Windows Setup
### 1. Open the project folder
Use PowerShell in the project root:

```powershell
cd C:\path\to\Final_Project
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel packaging
```

### 3. Install core project dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Install UI dependencies
```powershell
python -m pip install -r requirements-webui.txt
```

### 5. Install PyTorch and TorchCodec
Use the CUDA 12.8 wheel index so Windows and Linux use the same CUDA wheel family:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torchcodec==0.11.1 --index-url https://download.pytorch.org/whl/cu128
```

Expected versions after install:
- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `torchcodec 0.11.1+cu128`
- CUDA runtime reported by PyTorch: `12.8`

If your GPU driver cannot run CUDA 12.8 wheels, update the NVIDIA driver or install a PyTorch wheel family that matches your driver.

### 6. Install SAM2
If you already have a local `sam2/` source folder, install it in editable mode:

```powershell
cd sam2
python -m pip install -e .
cd ..
```

If you do not have `sam2/` yet, clone or download SAM2 first, then run the install command above.

### 7. Install SAM-Audio from the local source folder first
Use the project-local `sam-audio/` source folder first on Windows. From the project root:

```powershell
cd sam-audio
python -m pip install -e .
cd ..
```

Validate the import:

```powershell
python -c "import sam_audio; import sam_audio.model.base as b; print('sam_audio ok:', b.__file__)"
```

If the local `sam-audio/` install works and the pipeline runs, continue to the next section.

### 8. Windows fallback: clean clone, patch `base.py`, then install
Use this fallback if any of these happen:
- `sam-audio/` is missing
- `import sam_audio` fails
- SAM-Audio installs but gives runtime errors when running the project
- local model folders fail to load
- you see an error like `BaseModel._from_pretrained() missing 2 required keyword-only arguments: 'proxies' and 'resume_download'`

The important order is **clone first, patch `base.py` in the cloned repo, then install in editable mode**.

```powershell
cd C:\path\to\Final_Project

if (Test-Path sam-audio) {
    Rename-Item sam-audio sam-audio-broken
}

git clone https://github.com/facebookresearch/sam-audio.git sam-audio
notepad sam-audio\sam_audio\model\base.py
```

In `sam-audio\sam_audio\model\base.py`, patch `_from_pretrained()` using the code in [SAM-Audio `base.py` local model patch](#sam-audio-basepy-local-model-patch). Save the file, then install:

```powershell
cd sam-audio
python -m pip install -e .
cd ..

python -c "import sam_audio; import sam_audio.model.base as b; print('sam_audio ok:', b.__file__)"
```

If installing SAM-Audio changes your already-correct TorchCodec install, reinstall TorchCodec from the same PyTorch index:

```powershell
python -m pip install --force-reinstall torchcodec==0.11.1 --index-url https://download.pytorch.org/whl/cu128
```

### 9. Confirm ffmpeg
The project checks in this order:
1. explicit `--ffmpeg-bin` or UI override
2. `tools/ffmpeg/windows/ffmpeg.exe`
3. `tools/ffmpeg/linux/ffmpeg`
4. system `ffmpeg`

You can either:
- put `ffmpeg.exe` in `tools/ffmpeg/windows/ffmpeg.exe`, or
- install ffmpeg on Windows and add it to `PATH`

### 10. Hugging Face login
Required for real SAM-Audio inference and for downloading model assets. Paste your Hugging Face token only when prompted; do not put it in source files or commit it.

```powershell
python -m pip install -U "huggingface_hub[cli]"
hf auth login
```

### 11. Validate the Windows environment
```powershell
python -c "import cv2, numpy, scipy, pandas, PIL, matplotlib; print('base deps ok')"
python -c "import torch, torchvision, torchaudio, importlib.metadata as m; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('torchcodec', m.version('torchcodec')); print('cuda', torch.version.cuda)"
python -c "from torchcodec.decoders import AudioDecoder, VideoDecoder; print('torchcodec ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import sam_audio; print('sam_audio ok')"
python -c "from path_layout import resolve_ffmpeg_binary; print(resolve_ffmpeg_binary())"
```

## Linux Setup (Ubuntu + NVIDIA)
Linux setup uses a normal Python virtual environment. On shared Rutgers iLab machines, request a Slurm GPU allocation before running GPU jobs. For the full SAM2 + SAM-Audio pipeline, use a GPU with **40 GB VRAM or more** when possible. Smaller 16 GB or 24 GB GPUs may work for short or segmented runs, but they are much more likely to hit CUDA out-of-memory errors on iLab.

### Rutgers iLab rebuild path
On Rutgers iLab machines, a typical interactive GPU allocation is shown below. This requests one GPU and 40 GB of system RAM; VRAM depends on the GPU Slurm assigns, so prefer nodes with 40 GB+ GPUs such as A100-class machines when available.

```bash
srun -G 1 --mem=40g --pty bash
```

Then rebuild the environment from inside the project folder. This setup redirects pip, Hugging Face, PyTorch, and temp caches away from the default home cache locations, which helps avoid quota and temp-directory issues on shared systems.

```bash
cd /common/users/$USER/Final_Project

mkdir -p /common/users/$USER/pip-tmp
mkdir -p /common/users/$USER/pip-cache
mkdir -p /common/users/$USER/.cache
mkdir -p /common/users/$USER/hf-home
mkdir -p /common/users/$USER/torch-cache

export TMPDIR=/common/users/$USER/pip-tmp
export PIP_CACHE_DIR=/common/users/$USER/pip-cache
export XDG_CACHE_HOME=/common/users/$USER/.cache
export HF_HOME=/common/users/$USER/hf-home
export TORCH_HOME=/common/users/$USER/torch-cache

rm -rf ~/.cache/pip

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -r requirements-webui.txt
```

Install the known-working Linux CUDA 12.8 stack. Keeping PyTorch, TorchVision, TorchAudio, and TorchCodec on the same `cu128` wheel family avoids TorchCodec import errors such as missing `AudioDecoder`.

```bash
python -m pip install --no-cache-dir --force-reinstall \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --no-cache-dir --force-reinstall \
  torchcodec==0.11.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Expected versions after install:
- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `torchcodec 0.11.1+cu128`
- CUDA runtime reported by PyTorch: `12.8`

### 1. Install SAM2 after PyTorch
Use `--no-build-isolation` so pip does not create a separate temporary build environment.

```bash
cd /common/users/$USER/sam2
python -m pip install --no-cache-dir --no-build-isolation -e .
```

If the SAM2 CUDA extension causes install trouble, disable the CUDA extension and reinstall:

```bash
SAM2_BUILD_CUDA=0 python -m pip install --no-cache-dir --no-build-isolation -e .
```

Return to the project root:

```bash
cd /common/users/$USER/Final_Project
```

### 2. Install SAM-Audio from the local source folder first
Use the project-local `sam-audio/` source folder first on Linux. From the project root:

```bash
cd /common/users/$USER/Final_Project/sam-audio
python -m pip install --no-cache-dir -e .
cd /common/users/$USER/Final_Project
```

Validate the import:

```bash
python -c "import sam_audio; import sam_audio.model.base as b; print('sam_audio ok:', b.__file__)"
```

If the local `sam-audio/` install works and the pipeline runs, continue to Hugging Face login.

### 3. Linux fallback: clean clone, patch `base.py`, then install
Use this fallback if any of these happen:
- `sam-audio/` is missing
- `import sam_audio` fails
- SAM-Audio installs but gives runtime errors when running the project
- local model folders fail to load
- you see an error like `BaseModel._from_pretrained() missing 2 required keyword-only arguments: 'proxies' and 'resume_download'`

The important order is **clone first, patch `base.py` in the cloned repo, then install in editable mode**.

```bash
cd /common/users/$USER/Final_Project

if [ -d sam-audio ]; then
  mv sam-audio sam-audio-broken
fi

git clone https://github.com/facebookresearch/sam-audio.git sam-audio
nano sam-audio/sam_audio/model/base.py
```

In `sam-audio/sam_audio/model/base.py`, patch `_from_pretrained()` using the code in [SAM-Audio `base.py` local model patch](#sam-audio-basepy-local-model-patch). Save the file, then install:

```bash
cd sam-audio
python -m pip install --no-cache-dir -e .
cd ..

python -c "import sam_audio; import sam_audio.model.base as b; print('sam_audio ok:', b.__file__)"
```

If installing SAM-Audio changes your already-correct TorchCodec install, reinstall TorchCodec from the same PyTorch index:

```bash
python -m pip install --no-cache-dir --force-reinstall \
  torchcodec==0.11.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 4. Hugging Face login
Log into Hugging Face before real SAM-Audio inference. Paste the token only when prompted, and do not commit tokens to the repository.

```bash
python -m pip install -U "huggingface_hub[cli]"
hf auth login
```

### 5. Validate the Linux environment
```bash
python -c "import torch, torchvision, torchaudio, importlib.metadata as m; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('torchcodec', m.version('torchcodec')); print('cuda', torch.version.cuda)"
python -c "from torchcodec.decoders import AudioDecoder, VideoDecoder; print('torchcodec ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import sam_audio; print('sam_audio ok')"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python -c "from path_layout import resolve_ffmpeg_binary; print(resolve_ffmpeg_binary())"
```

### 6. Launch the WebUI on Linux
```bash
streamlit run webui.py
```

### Generic Linux venv path
For a non-iLab Linux machine, use the same venv-based install flow from the project root:

```bash
cd /path/to/Final_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install -r requirements.txt
python -m pip install -r requirements-webui.txt
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torchcodec==0.11.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install local source checkouts:

```bash
cd sam2
python -m pip install --no-build-isolation -e .
cd ..

cd sam-audio
python -m pip install -e .
cd ..
```

If the local SAM-Audio checkout does not work, use the same Linux fallback flow above: clean clone `sam-audio/`, patch `sam-audio/sam_audio/model/base.py`, then install with `python -m pip install -e .`.

Finish with Hugging Face login and the validation commands from the Linux section above.

## SAM-Audio Notes
This README is the single setup reference for SAM-Audio. The project expects the official `sam_audio` package for real visual-prompted audio separation, installed from the project-local `sam-audio/` source folder when possible.

Required pieces:
- Python environment with `torch`, `torchaudio`, `torchcodec`, `opencv-python`, `numpy`, and `ffmpeg`
- Official SAM-Audio code installed so `from sam_audio import SAMAudio, SAMAudioProcessor` works
- Hugging Face authentication with access to the SAM-Audio model repo or local model folders
- Matching CUDA wheel family across PyTorch, TorchVision, TorchAudio, and TorchCodec

Recommended model ids when downloading from Hugging Face:
- `facebook/sam-audio-small-tv`
- `facebook/sam-audio-base-tv`
- `facebook/sam-audio-large-tv`

The `-tv` variants are the recommended defaults for this project because the pipeline uses visual prompting from tracker masks.

For portable local runs, keep SAM-Audio models under `sam_audio_models/` in the project root:
- `sam_audio_models/small-tv/`
- `sam_audio_models/base-tv/`
- `sam_audio_models/large-tv/`

When using a local model folder, the UI and CLI should point to relative model ids like:
- `sam_audio_models/small-tv`
- `sam_audio_models/base-tv`

Keep source checkouts at:
- `sam2/`
- `sam-audio/`

This keeps local imports predictable across Windows and Linux.

## SAM-Audio base.py local model patch
If SAM-Audio fails to load local model folders or raises an error like `BaseModel._from_pretrained() missing 2 required keyword-only arguments: 'proxies' and 'resume_download'`, patch the active SAM-Audio source file.

Patch targets:
- Windows project source: `sam-audio\sam_audio\model\base.py`
- Linux project source: `sam-audio/sam_audio/model/base.py`
- Installed package fallback: print the active path with `python -c "import sam_audio.model.base as b; print(b.__file__)"`

When using the fallback flow, patch the freshly cloned source file **before** running `python -m pip install -e .`.

Inside `class BaseModel`, update `_from_pretrained()` so it accepts the newer Hugging Face arguments and uses local model folders directly when `model_id` is a directory:

```python
@classmethod
def _from_pretrained(
    cls,
    *,
    model_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    resume_download: bool = False,
    local_files_only: bool = False,
    token: Union[str, bool, None] = None,
    map_location: str = "cpu",
    strict: bool = True,
    revision: Optional[str] = None,
    **model_kwargs,
):
    if os.path.isdir(model_id):
        cached_model_dir = model_id
    else:
        cached_model_dir = snapshot_download(
            repo_id=model_id,
            revision=revision or getattr(cls, "revision", None),
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
```

Keep the rest of the function that reads `config.json`, loads `checkpoint.pt`, and calls `model.load_state_dict(...)`.

Make sure the file already imports the names used above. If it does not, add these near the top of `base.py`:

```python
import os
from typing import Dict, Optional, Union
from huggingface_hub import snapshot_download
```

Use spaces only for indentation. Mixing tabs and spaces can cause `TabError` on Linux.

After patching and installing, validate the import:

```bash
python -c "import sam_audio; import sam_audio.model.base as b; print('sam_audio ok:', b.__file__)"
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

```bash
streamlit run webui.py
```

Open the local URL that Streamlit prints in the terminal.

## Running The Pipeline From Terminal
### 1. Tracker only
Interactive point selection example on Windows:

```powershell
python tracker.py input\test1_TomScott.mp4 output\terminal_runs\tracker_test1\video\tracked.mp4 --checkpoint checkpoints\sam2.1_hiera_tiny.pt --config configs\sam2.1\sam2.1_hiera_t.yaml --frames-dir output\terminal_runs\tracker_test1\frames --artifacts-dir output\terminal_runs\tracker_test1\artifacts --mode point --preview
```

Linux equivalent:

```bash
python tracker.py input/test1_TomScott.mp4 output/terminal_runs/tracker_test1/video/tracked.mp4 --checkpoint checkpoints/sam2.1_hiera_tiny.pt --config configs/sam2.1/sam2.1_hiera_t.yaml --frames-dir output/terminal_runs/tracker_test1/frames --artifacts-dir output/terminal_runs/tracker_test1/artifacts --mode point --preview
```

### 2. Audio only
Use tracker masks plus the matching video segment.

Windows:

```powershell
python audio_pipeline.py --video-path input\test1_TomScott.mp4 --mask-path output\terminal_runs\tracker_test1\artifacts\masks\masks.npz --output-dir output\terminal_runs\audio_test1 --clip-id test1_TomScott --model-id sam_audio_models\small-tv --predict-spans --reranking-candidates 1
```

Linux:

```bash
python audio_pipeline.py --video-path input/test1_TomScott.mp4 --mask-path output/terminal_runs/tracker_test1/artifacts/masks/masks.npz --output-dir output/terminal_runs/audio_test1 --clip-id test1_TomScott --model-id sam_audio_models/small-tv --predict-spans --reranking-candidates 1
```

### 3. Evaluation
If you have clean reference audio and/or ground-truth masks, run evaluation.

Windows:

```powershell
python evaluation.py --output-dir output\eval_test1 --clip-id test1_TomScott --model-id sam_audio_models\small-tv --predicted-mask-path output\terminal_runs\tracker_test1\artifacts\masks\masks.npz --estimated-audio-path output\terminal_runs\audio_test1\target.wav --reference-audio-path input\reference.wav --ground-truth-mask-path input\gt_masks.npz --audio-metadata-path output\terminal_runs\audio_test1\audio_run_metadata.json
```

Linux:

```bash
python evaluation.py --output-dir output/eval_test1 --clip-id test1_TomScott --model-id sam_audio_models/small-tv --predicted-mask-path output/terminal_runs/tracker_test1/artifacts/masks/masks.npz --estimated-audio-path output/terminal_runs/audio_test1/target.wav --reference-audio-path input/reference.wav --ground-truth-mask-path input/gt_masks.npz --audio-metadata-path output/terminal_runs/audio_test1/audio_run_metadata.json
```

### 4. Analysis
Windows:

```powershell
python analyze_results.py output\ui_runs\SOME_JOB_ID\runs --output-dir output\analysis_test1
```

Linux:

```bash
python analyze_results.py output/ui_runs/SOME_JOB_ID/runs --output-dir output/analysis_test1
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
