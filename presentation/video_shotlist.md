# Demo Video Shot List — NeuroRL (60–90 seconds)

**Setup before recording:**
- Browser open to https://abhishekbiradar-neuro-rl-env.hf.space/docs
- VS Code open with `scripts/live_demo.py` visible in left pane
- Terminal in right pane, ready to run the script
- `HF_USER=abhishekBiradar` exported in shell
- Resolution: 1920×1080, font size 14+ in terminal

---

## 0:00 – 0:10 · Title card

**Screen:** Slide 1 from `presentation/slides.md` (Marp or browser fullscreen)

**Narration:**
> "700,000 people worldwide are locked in — unable to move or speak — but fully conscious.
> Brain-computer interfaces can restore their voice. The problem is that the decoder
> degrades silently as the brain changes. This is NeuroRL."

**Notes:** Hold on the title for the full 10 seconds. Fade-in if your recording software supports it.

---

## 0:10 – 0:25 · Live Space API walkthrough

**Screen:** Browser at `https://abhishekbiradar-neuro-rl-env.hf.space/docs`

**Actions:**
1. Scroll to show `/reset` and `/step` endpoints in the Swagger UI.
2. Click `POST /reset` → "Try it out" → "Execute".
   Show the returned JSON: spike matrix, mean firing rates, drift phase.
3. Briefly point at the `drift_phase` field — *"this is the non-stationarity the agent must track"*.

**Narration:**
> "The environment is a live HuggingFace Space — a stateless HTTP API.
> A POST to /reset returns a 20-channel spike train with a drift phase baked in.
> The agent posts its decoded JSON action to /step and gets back a reward."

**Notes:** Don't execute /step here (needs a valid action body). Just show the Swagger UI response from /reset.

---

## 0:25 – 0:50 · VS Code split — live_demo.py running

**Screen:** VS Code split view
- **Left pane:** `scripts/live_demo.py` source, scrolled to the main loop (~line 130)
- **Right pane:** integrated terminal running the script

**Actions:**
1. Point at the model-load section briefly: *"loading Qwen3-1.7B with the trained LoRA adapter"*.
2. Switch focus to terminal. Type and run:
   ```
   python scripts/live_demo.py
   ```
3. Let at least 5–8 episodes stream through. Show:
   - The coloured intent box (green = correct, red = wrong)
   - The `<think>` reasoning snippet
   - The reward value per step
4. Glance at the matplotlib window updating on the right — rolling accuracy line ticking up,
   drift phase dots forming a curve.

**Narration:**
> "The trained agent reasons about which neurons are firing above baseline,
> cites the discriminative channels, and emits a structured JSON decode.
> Green means it nailed the intent — watch the rolling accuracy climb above the random-guess line."

**Notes:**
- If inference is slow on CPU, record only 3–5 episodes and speed up post-production.
- The matplotlib window should be visible in screen share. Alt-tab briefly to show it if it
  went behind VS Code.

---

## 0:50 – 1:10 · Three plots side by side

**Screen:** Switch to a browser or image viewer showing all three PNGs side by side
(or use Slide 7 from `presentation/slides.md`)

**Actions:**
1. Point at the reward curve: *"reward rises from near-zero as the agent learns the rubric"*.
2. Point at the accuracy bar chart: *"trained model — TODO:XX% — vs 14% random guess"*.
3. Point at the drift resistance plot: *"across a full 2π drift cycle, performance stays above the 80% clinical target"*.

**Narration:**
> "After GRPO fine-tuning: TODO:XX% held-out accuracy versus 14% for the untrained model.
> And critically — the trained agent stays above the 80% clinical target across almost
> the entire drift cycle. The static decoder never gets off the floor."

**Notes:** Fill in `TODO:XX%` with the real number from `outputs/trained_metrics.json` before recording.

---

## 1:10 – 1:30 · Final slide + thank-you

**Screen:** Slide 8 from `presentation/slides.md`

**Actions:**
1. Linger on the bold accuracy numbers.
2. Scroll to the "What's next" bullets — mention DRPO and real BCI data briefly.
3. Hold on the links at the bottom (GitHub, HF Space, adapter).

**Narration:**
> "A patient who wakes up Thursday with a shifted neural code can still use their device —
> without waiting for a Friday clinic appointment. The code, adapter, and live environment
> are all public. Thank you."

**Notes:** Fade to black or hold on the slide for the last 5 seconds of the recording.

---

## Post-production checklist

- [ ] Trim dead air between actions
- [ ] Replace all `TODO:XX%` placeholders with real numbers before final export
- [ ] Add captions/subtitles if submitting to accessibility-aware venues
- [ ] Export at 1080p, H.264, ≤ 200 MB
- [ ] Upload to YouTube (unlisted or public) and paste URL into README + blog post
