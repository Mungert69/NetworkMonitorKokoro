# Model Attribution and Licenses

This project uses third-party text-to-speech (TTS) and speech-to-text (ASR)
models. The license in [LICENSE](LICENSE) covers the software in this
repository, not the third-party model weights or voices. Each model remains
subject to its own terms.

This file records provenance and licensing information to the best of our
knowledge. Verify the current terms with the upstream sources before
redistributing or using a model commercially.

## Models vendored in this repository

### HAL 9000 (no denoise) — English (EN), Medium

- **Files:** `kokoro_model/piper_repo/alt-hal.onnx`,
  `kokoro_model/piper_repo/alt-hal.onnx.json`
- **Voice creator:** dividebysandwich
- **Catalog listing:** <https://piperhub.org/model/hal-9000-no-denoise-english-en-medium/>
  (retrieved 2026-10-22)
- **Distribution repository:** <https://huggingface.co/fox3000foxy/piper-voices-collection>
  (path: `dividebysandwich--piper-voice-models/HAL9000-no-denoise_en_US-hal_12894-medium.onnx.onnx`)
- **License declared by publisher:** **Not specified.** The Hugging Face
  repository has no license metadata, and the PiperHub listing states
  "License: not specified".
- **Usage note:** This voice is included for personal and experimental
  evaluation only. It is **not** represented as cleared for redistribution or
  commercial use. Contact the voice creator/uploader for licensing terms. If
  you are a rights holder and want this file removed, please open an issue.
- **Technical details:** Piper voice format (`piper_version` 1.0.0), 1 speaker,
  `en-us` phonemes, 22,050 Hz, medium quality.

### HAL 9000 (campwill) — default Piper voice

- **Files:** downloaded at runtime to `kokoro_model/piper_repo/hal.onnx` and
  `hal.onnx.json`; not vendored in this repository.
- **Author:** campwill
- **Source:** <https://huggingface.co/campwill/HAL-9000-Piper-TTS>
- **License:** Apache-2.0 (per the model card and repository metadata)
- **Training data:** <https://huggingface.co/datasets/campwill/HAL-9000-Speech>,
  audio collected from *2001: A Space Odyssey*
- **Usage note:** The model weights are published under Apache-2.0, but the
  underlying film audio and the HAL 9000 character/likeness may be subject to
  separate rights held by third parties.

## Models downloaded at runtime

### Kokoro-82M v1.0 (ONNX)

- **Files:** `kokoro_model/onnx/model.onnx`, `kokoro_model/voices/<voice>.bin`,
  and related config files
- **Upstream model:** <https://huggingface.co/hexgrad/Kokoro-82M> (Apache-2.0)
- **ONNX conversion:** <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX>
  (Apache-2.0)
- **Default voice:** `am_adam`

### Moonshine (ASR)

- **Default engine:** Moonshine Medium Streaming (`moonshine-voice` 0.1.5,
  MIT per package metadata)
- **Model source:** downloaded from `download.moonshine.ai` and cached under
  `~/.cache/moonshine_voice`
- **Upstream project:** <https://github.com/moonshine-ai/moonshine>
- **Note:** check the upstream model terms for the downloaded weights.

### Wav2Vec2 (optional ASR)

- **Models:** `facebook/wav2vec2-base-960h` and
  `onnx-community/wav2vec2-base-960h-ONNX`
- **License:** Apache-2.0

### Silero VAD

- **Package:** `silero-vad` 6.2.3
- **License:** MIT
- **Source:** <https://github.com/snakers4/silero-vad>

## Runtime components relevant to model use

- **Piper CLI:** `piper-tts` 1.8.0 from
  <https://github.com/OHF-Voice/piper1-gpl> — GPL-3.0-or-later (per package
  metadata). The original <https://github.com/rhasspy/piper> project is MIT.
- **espeak-ng:** used by Piper for phonemization — GPL-3.0-or-later.
- **ONNX Runtime:** MIT.

## Notes

- Model files may be large; the vendored ONNX file is stored using Git LFS.
- If you redistribute this repository, keep this file and comply with each
  model's terms.
- Corrections and attribution updates are welcome via an issue or pull request.
