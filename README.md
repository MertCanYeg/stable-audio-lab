# Stable Audio Lab

A modern, reproducible, and portable workspace for **Stable Audio 3** (`small-music`, `small-sfx`, and `medium`) powered by **Pixi**.

Designed to run locally with NVIDIA GPU acceleration (via PyTorch + CUDA) or CPU fallback, with zero global environment pollution.

---

## ⚡ Highlights

- **Model Family**: Stability AI's **Stable Audio 3**
  - `small-music`: Stereo composition and music generation (up to 120s)
  - `small-sfx`: Sound effects and ambient audio generation (up to 120s)
  - `medium`: Unified flagship quality model for cinematic music and sound design (up to 380s)
- **100% Declarative & Portable**: Fully managed by [Pixi](https://pixi.sh). A single `pixi.lock` file guarantees identical builds on any computer.
- **Hardware Accelerated**: Automatically utilizes NVIDIA GPUs (Tensor Cores / FP16) when available, while seamlessly supporting CPU fallback.
- **Strict KISS & YAGNI Design**: Lean, flat, and professional-grade codebase with zero unnecessary boilerplate or wrappers.
- **Flexible Interfaces**:
  - **CLI**: Fast scriptable generation (`pixi run generate`)
  - **Web UI**: Interactive Gradio studio with waveforms and presets (`pixi run ui`)
  - **Diagnostics**: Built-in environment & GPU verification (`pixi run check-env`)
  - **Download Manager**: Cache inspection and pre-download manager (`pixi run download-models`)

---

## 📋 Prerequisites

1. **Pixi**: If not installed, install it in seconds:
   - **Windows (PowerShell)**: `iwr -useb https://pixi.sh/install.ps1 | iex`
   - **Linux/macOS**: `curl -fsSL https://pixi.sh/install.sh | bash`
2. **Hugging Face Model Access**:
   Stable Audio 3 weights are community-licensed and gated on Hugging Face:
   1. Log in to [Hugging Face](https://huggingface.co).
   2. Accept the model license terms:
      - [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
      - [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
      - [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
   3. Create an Access Token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   4. Copy `.env.example` to `.env` and paste your token:
      ```bash
      cp .env.example .env
      ```

---

## 🚀 Quickstart

### 1. Verify Environment & GPU
Check that PyTorch detects your GPU and verifies Hugging Face authentication:
```bash
pixi run check-env
```

### 2. Run Automated Test Suite
Verify that all core modules, prompt sanitization, model registries, and storage validators are operational:
```bash
pixi run test
```

### 3. Generate Audio via CLI
Generate high-fidelity stereo audio in seconds:
```bash
# Music generation (small-music)
pixi run generate --prompt "Upbeat funky bassline with warm rhodes piano and crisp drums" --duration 15

# Sound effect generation (small-sfx)
pixi run generate --model small-sfx --prompt "Campfire crackling in a dense forest with gentle wind" --duration 10

# High-fidelity music and sound design (medium)
pixi run generate --model medium --prompt "An epic cinematic orchestral trailer theme with soaring strings" --duration 15
```

All generated `.wav` files are automatically timestamped, slugified, and saved to the `outputs/` directory in standard 16-bit PCM CD-quality format. The exact seed used is displayed in the terminal so you can reproduce any track.

### 4. Launch Interactive Web Studio
Start the local Gradio studio in your web browser:
```bash
pixi run ui
```
Open `http://127.0.0.1:7860` in your browser.

---

## 🛠️ CLI Options (`generate.py`)

| Flag | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--prompt` | `-p` | *required* | Text description of the audio |
| `--negative-prompt`| `-n` | `None` | Qualities or sounds to avoid |
| `--model` | `-m` | `small-music` | Model variant: `small-music`, `small-sfx`, or `medium` |
| `--duration` | `-d` | `15.0` | Output duration in seconds (Small: up to 120s, Medium: up to 380s) |
| `--steps` | `-s` | `8` | Diffusion sampling steps (8 is optimal for post-trained models) |
| `--cfg-scale` | | `1.0` | Classifier-free guidance scale |
| `--seed` | | `-1` | Seed for reproducibility (`-1` for random; seed is logged) |
| `--output` | `-o` | `None` | Custom path to save the output `.wav` file |

---

## 📦 Pre-Downloading Models (Optional)

If you prefer to inspect or pre-download model weights from the terminal:

```bash
# Check cache status of all models
pixi run download-models --status

# Pre-download all models
pixi run download-models

# Pre-download a specific model
pixi run download-models --model medium
```

Within the Gradio Web Studio (`pixi run ui`), the interface displays a live **Model Status** badge and a one-click download button.

---

## 📁 Project Structure

```
stable-audio-lab/
├── core/                  # Clean, modular engine & utilities
│   ├── compat.py          # Platform & environment setup (UTF-8 stdout, warning filters, Triton fallback)
│   ├── registry.py        # Model specifications, durations, and curated presets
│   ├── storage.py         # Hugging Face Hub caching and download helpers
│   └── engine.py          # VRAM cache hygiene, reproducible inference & 16-bit PCM WAV export
├── tests/                 # Automated unit tests (slugify, registry, compat, storage, validation)
├── app.py                 # Gradio Web Studio with unified model selector & native progress
├── generate.py            # Lean CLI generation tool with tqdm progress & seed capture
├── download_models.py     # Terminal model cache inspector & pre-download tool
├── check_env.py           # Hardware & authentication diagnostic script
├── pixi.toml              # Declarative environment & tasks definition
├── pixi.lock              # Cross-platform dependency lockfile
├── LICENSE                # MIT License
├── .env.example           # Template for Hugging Face access token
└── outputs/               # Destination folder for generated audio files
```

---

## 💻 Running on Another Computer

To run this project on another computer:
```bash
git clone https://github.com/MertCanYeg/stable-audio-lab.git
cd stable-audio-lab
cp .env.example .env   # add your HF_TOKEN
pixi run ui            # automatically installs environment & launches studio
```
To pull the latest updates anytime:
```bash
pixi run update        # pulls latest changes from repository
```
Pixi automatically handles Python installation, CUDA runtimes, PyTorch, and all dependencies without affecting the host system.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Model weights are provided by Stability AI under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-medium).
