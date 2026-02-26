# ASR Agent

An agent skill for Automatic Speech Recognition using NVIDIA NeMo models.

## Features

- Single-file transcription with pre-trained Parakeet / Canary models
- Batch transcription from a directory or manifest
- Word-level timestamps
- Multilingual support via Canary (EN, ES, FR, DE)
- Output in plain text, JSON, or SRT subtitle format

## Prerequisites

```bash
pip install "nemo_toolkit[asr]"
pip install soundfile  # for audio handling
```

## Usage

### Transcribe a Single File

```bash
python agents/asr-agent/transcribe.py \
  --audio path/to/audio.wav \
  --model nvidia/parakeet-ctc-1.1b \
  --output transcript.txt
```

### Batch Transcription

```bash
python agents/asr-agent/batch_transcribe.py \
  --input-dir audio/ \
  --output transcripts.json \
  --model nvidia/parakeet-ctc-1.1b \
  --batch-size 8
```

### Multilingual Transcription

```bash
python agents/asr-agent/transcribe.py \
  --audio spanish_audio.wav \
  --model nvidia/canary-1b \
  --source-lang es \
  --target-lang en \
  --output translated_transcript.txt
```

### Generate SRT Subtitles

```bash
python agents/asr-agent/transcribe.py \
  --audio video_audio.wav \
  --model nvidia/parakeet-ctc-1.1b \
  --output-format srt \
  --output subtitles.srt
```

## Models Reference

| Model | Best For | GPU Memory |
|-------|----------|------------|
| `nvidia/parakeet-ctc-1.1b` | Highest English accuracy | ~6GB |
| `nvidia/parakeet-rnnt-1.1b` | Streaming + accuracy | ~6GB |
| `nvidia/parakeet-tdt-1.1b` | Speed + accuracy balance | ~6GB |
| `nvidia/canary-1b` | Multilingual + translation | ~6GB |

## Prompts

See [docs/PROMPTS.md](../../docs/PROMPTS.md#asr-prompts) for post-processing prompt templates.
