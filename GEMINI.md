# Stable Audio Lab - Project Guidelines & Rules

## 1. Strict Invariant: Zero Arbitrary Values
- **NEVER use arbitrary, speculative, or hardcoded approximations**:
  - Do NOT invent estimated wait times (e.g., `~9s`, `~10-15s`).
  - Do NOT invent approximate VRAM numbers (e.g., `~2.5 GB`, `~6.5 GB`).
  - Do NOT invent artificial progress percentage splits (e.g., `0.15 + 0.75 * step / total`).
  - Do NOT invent model marketing names or speculative parameter tags (e.g., `"Music (433M)"`, `"Cinema & Production (1.4B)"`). Always use verified official model names (`small-music`, `small-sfx`, `medium`).
- **All reported values must be strictly factual or live-measured**:
  - Specifications (parameters, disk size, max duration, sample rate) must come directly from verified model configs.
  - Telemetry (VRAM usage, step counts, execution times) must be measured live via PyTorch (`torch.cuda.memory_allocated()`) or system timers (`time.time()`).
  - **Isolated Diffusion Step Timing**: Diffusion iteration speed (`steps/s`), elapsed sampling time, and ETAs must strictly measure the active diffusion loop. Never contaminate sampling metrics with pre-conditioning duration (e.g., T5 prompt encoding) or post-sampling VAE latent decoding.
  - **Gradio Progress Tracking**: Always pass step tuples `progress((current_step, total_steps), desc=..., unit="steps")` to display live iteration counts, steps/s, and ETAs in the UI. Never pass bare `float` percentages during iterative sampling (which drops iteration metrics and causes generic `"processing | X.Xs"` fallback badges). Non-step stages (e.g. VRAM loading, tokenization) must use indeterminate status `progress(None, desc=...)`.

## 2. Professional UI/UX Standards
- **No "Fake Simplicity"**: Minimal code does not mean degraded user experience.
- **Dual Aspect Ratio Verification (16:9 & 8:9)**: When verifying UI visually via screenshots, always test across both standard full-size **16:9** (1920x1080) and snapped split-screen **8:9** (960x1080) viewports. Ensure controls, sliders, and telemetry remain readable and unclipped across both.
- **Tab Height Invariance & Zero Clipping**: Sliders, min/max numbers, and input labels must never be clipped or wrap unexpectedly. Textboxes, banners, and action controls must maintain identical pixel heights across all tabs to prevent vertical shifting during tab navigation.
- **Gradio `track_tqdm=False` Mandatory**: Always set `track_tqdm=False` on `gr.Progress()`. Never allow Gradio to monkeypatch `tqdm`. Doing so redirects console progress to `os.devnull`, captures internal third-party loops, and causes phantom duplicate bars and double-counted percentages (`175%`, `14/8 steps`).
- **In-Place Terminal Progress (`tqdm`)**: Use a single dedicated `tqdm` bar updating in-place on a single line via `\r`. NEVER call `print()` inside a sampling callback or step loop, as `print()` emits newlines (`\n`) and breaks `tqdm`'s in-place carriage returns.
- **Silence Third-Party Progress Noise**: Proactively silence transient internal library progress bars (e.g., `transformers` loading cached weight tensors) via `transformers.utils.logging.disable_progress_bar()` and `HF_HUB_DISABLE_PROGRESS_BARS=1`.
- **Prompt Examples**: Always present prompt templates in non-paginated, smoothly scrollable tables (`examples_per_page=100`, CSS `overflow-y: auto`). Never use paginated numbered buttons.
- **Immediate User Feedback**: Always acknowledge user actions immediately via `progress(None, desc=...)` at the start of execution. Never leave users waiting without visual updates.
- **Model Transparency**: Each model tab must clearly display its exact specifications (parameters, disk size, max duration, audio format), context-aware negative prompt guidance, and a model-specific action button.
- **Error Recovery**: Catch domain exceptions and OOM errors cleanly. Clear CUDA memory and display actionable error toasts rather than crashing.

## 3. Engineering Principles (KISS, DRY, YAGNI)
- **Rely on Native Framework Primitives**:
  - Do NOT build custom worker threads, queues, or polling loops around Gradio; Gradio's internal threadpool, `gr.Progress()`, and `gr.Error()` handle concurrency and progress natively.
  - Do NOT build complex generator `yield` streaming wrappers when standard return values `(audio_file, status_msg)` coupled with `gr.Progress()` provide superior stability and simplicity.
  - Do NOT add wrapper layers, callback filter chains, or monkeypatches unless strictly necessary.
- **Clean Architecture**:
  - Encapsulate parameters in typed dataclasses (`GenerationConfig`) with explicit validation.
  - Maintain a clean domain exception hierarchy rooted at `StableAudioError`.
  - Use `pathlib.Path` consistently across all modules.
  - Reconfigure Windows stdout/stderr to UTF-8 (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`) on import to prevent fatal `charmap` encoding errors with progress characters and emojis.
