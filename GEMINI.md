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
- **Inline Typography & Badge Baseline Alignment**: Never apply `display: flex` or `align-items: center` directly to text paragraphs (`<p>`) containing inline badges (`<code>`), bold tags, or text nodes. Flex layout collapses whitespace between inline text and breaks typographic baselines. Paragraphs must remain standard text flow (`display: block`), while inline `<code>` badges must remain compact (`font-size: 0.85em`, `padding: 1px 3px`, `margin: 0`, `vertical-align: 0px`) and align with the surrounding text baseline.
- **Form Checkbox Alignment**: In Gradio form groups, explicitly anchor checkboxes flush to the left edge (`justify-content: flex-start !important; width: 100% !important; margin-right: auto !important;`) rather than allowing default centered alignment.
- **Action Button Standards**: Primary and secondary action buttons (`Generate`, `Clear Output`) must maintain standard `1rem` text size and `48px` minimum height across all tabs.
- **Gradio `track_tqdm=False` Mandatory**: Always set `track_tqdm=False` on `gr.Progress()`. Never allow Gradio to monkeypatch `tqdm`. Doing so redirects console progress to `os.devnull`, captures internal third-party loops, and causes phantom duplicate bars and double-counted percentages (`175%`, `14/8 steps`).
- **In-Place Terminal Progress (`tqdm`)**: Use a single dedicated `tqdm` bar updating in-place on a single line via `\r`. NEVER call `print()` inside a sampling callback or step loop, as `print()` emits newlines (`\n`) and breaks `tqdm`'s in-place carriage returns.
- **Silence Third-Party Progress Noise**: Proactively silence transient internal library progress bars (e.g., `transformers` loading cached weight tensors) via `transformers.utils.logging.disable_progress_bar()` and `HF_HUB_DISABLE_PROGRESS_BARS=1`.
- **Prompt Examples**: Always present prompt templates in non-paginated, smoothly scrollable tables (`examples_per_page=100`, CSS `overflow-y: auto`). Never use paginated numbered buttons.
- **Immediate User Feedback**: Always acknowledge user actions immediately via `progress(None, desc=...)` at the start of execution. Never leave users waiting without visual updates.
- **Model Transparency**: Each model tab must clearly display its exact specifications (parameters, disk size, max duration, audio format), context-aware negative prompt guidance, and a model-specific action button.
- **Error Recovery**: Catch domain exceptions and OOM errors cleanly. Clear CUDA memory and display actionable error toasts rather than crashing.

## 3. Engineering Principles (KISS, DRY, YAGNI)
- **Urgent Targeted Fix Execution**: When the user requests a quick fix with instructions like *"fix this quickly, do not waste time"*, bypass lengthy headless browser capture chains or background diagnostic test scripts. Directly apply the targeted code fix, verify unit tests, and report back immediately.
- **Rely on Native Framework Primitives**:
  - Do NOT build custom worker threads, queues, or polling loops around Gradio; Gradio's internal threadpool, `gr.Progress()`, and `gr.Error()` handle concurrency and progress natively.
  - Do NOT build complex generator `yield` streaming wrappers when standard return values `(audio_file, status_msg)` coupled with `gr.Progress()` provide superior stability and simplicity.
  - Do NOT add wrapper layers, callback filter chains, or monkeypatches unless strictly necessary.
- **Clean Architecture**:
  - Encapsulate parameters in typed dataclasses (`GenerationConfig`) with explicit validation.
  - Maintain a clean domain exception hierarchy rooted at `StableAudioError`.
  - Use `pathlib.Path` consistently across all modules.
  - Reconfigure Windows stdout/stderr to UTF-8 (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`) on import to prevent fatal `charmap` encoding errors with progress characters and emojis.
