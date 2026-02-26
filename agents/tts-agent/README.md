# TTS Agent

An agent skill for Text-to-Speech synthesis using NVIDIA NeMo models.

## Features

- High-quality English speech synthesis with FastPitch + HiFi-GAN
- Text preparation and normalization for optimal synthesis
- Batch generation from text files
- WAV output (24-bit, 22kHz)

## Prerequisites

```bash
pip install "nemo_toolkit[tts]"
pip install soundfile
```

## Usage

### Synthesize Speech from Text

```bash
python agents/tts-agent/synthesize.py \
  --text "Hello, welcome to NVIDIA NeMo!" \
  --output speech.wav
```

### Synthesize from a File

```bash
python agents/tts-agent/synthesize.py \
  --input-file article.txt \
  --output narration.wav
```

### Batch Synthesis from Multiple Sentences

```bash
python agents/tts-agent/batch_synthesize.py \
  --input sentences.txt \
  --output-dir audio_output/
```

## Models Reference

| Acoustic Model | Vocoder | Quality |
|----------------|---------|---------|
| `nvidia/tts_en_fastpitch` | `nvidia/tts_hifigan` | High quality (recommended) |

## Prompts

See [docs/PROMPTS.md](../../docs/PROMPTS.md#tts-prompts) for text preparation prompt templates.
