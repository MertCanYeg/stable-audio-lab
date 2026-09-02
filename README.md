# Stable Audio Lab

A modern, reproducible, and portable workspace for **Stable Audio 3 Small** (`small-music` and `small-sfx`) powered by **Pixi**.

Designed to run locally with NVIDIA GPU acceleration (via PyTorch + CUDA) or CPU fallback, with zero global environment pollution.

---

## ⚡ Highlights

- **Model Family**: Stability AI's **Stable Audio 3 Small** (433M parameters)
  - `small-music`: Composition and music generation (up to 120s)
  - `small-sfx`: Sound effects and ambient audio generation (up to 120s)
- **100% Declarative & Portable**: Fully managed by [Pixi](https://pixi.sh). A single `pixi.lock` file guarantees identical builds on any computer.
- **Hardware Accelerated**: Automatically utilizes NVIDIA GPUs (Tensor Cores / FP16) when available, while seamlessly supporting CPU fallback.
- **Flexible Interfaces**:
  - **CLI**: Fast scriptable generation (`pixi run generate`)
  - **Web UI**: Interactive Gradio app with waveforms and spectrograms (`pixi run ui`)
  - **Diagnostics**: Built-in environment & GPU verification (`pixi run check-env`)

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

### 2. Generate Audio via CLI
Generate high-fidelity stereo audio in seconds:
```bash
# Music generation (small-music)
pixi run generate --prompt "Upbeat funky bassline with warm rhodes piano and crisp drums" --duration 15

# Sound effect generation (small-sfx)
pixi run generate --model small-sfx --prompt "Campfire crackling in a dense forest with gentle wind" --duration 10

# High-fidelity flagship music (medium - 1.4B parameters)
pixi run generate --model medium --prompt "An epic cinematic orchestral trailer theme with soaring strings" --duration 15
```

All generated `.wav` files are automatically timestamped and saved to the `outputs/` directory.

### 3. Launch Interactive Web Studio
Start the local Gradio studio in your web browser:
```bash
# Default: Launches with switchable tabs (🎵 Music + 🔊 Sound Effects + 🎛️ Medium + ℹ️ Diagnostics)
pixi run ui

# Single-model mode: Launch exclusively with a single model
pixi run ui --model small-music
pixi run ui --model small-sfx
pixi run ui --model medium
```
Open `http://127.0.0.1:7860` in your browser.

> 💡 **Memory Management:**
> On Windows (WDDM), the NVIDIA driver automatically provides Shared GPU Memory from system RAM. If a generation requires more memory than dedicated physical VRAM, the driver transparently pages into system RAM rather than raising an out-of-memory error.

---

## 🛠️ CLI Options (`generate.py`)

| Flag | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--prompt` | `-p` | *required* | Text description of the audio |
| `--negative-prompt`| `-n` | `None` | Qualities or sounds to avoid |
| `--model` | `-m` | `small-music` | Model variant: `small-music`, `small-sfx`, or `medium` |
| `--duration` | `-d` | `30.0` | Output duration in seconds (Small: up to 120s, Medium: up to 380s) |
| `--steps` | `-s` | `8` | Diffusion sampling steps (8 is optimal for post-trained models) |
| `--seed` | | `-1` | Seed for reproducibility (`-1` for random) |
| `--output` | `-o` | `None` | Custom path to save the output `.wav` file |
| `--no-half` | | `False` | Force float32 precision instead of fp16 |

---

## 📁 Project Structure

```
stable-audio-lab/
├── pixi.toml          # Declarative environment & tasks definition
├── pixi.lock          # Cross-platform dependency lockfile
├── check_env.py       # Diagnostic script for GPU, PyTorch, and HF Auth
├── generate.py        # CLI generation tool
├── app.py             # Gradio web interface launcher
├── LICENSE            # MIT License
├── .env.example       # Template for Hugging Face access token
├── .gitignore         # Ignores outputs/, .env, and build caches
└── outputs/           # Destination folder for generated audio files
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
Pixi automatically handles Python installation, CUDA runtimes, PyTorch, and all dependencies without affecting the host system.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Model weights are provided by Stability AI under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-medium).
