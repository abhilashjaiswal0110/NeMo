# Audio Agent

An agent skill for audio processing and analysis using NVIDIA NeMo models.

## Features

- Speech enhancement and noise reduction
- Speaker diarization (who spoke when)
- Audio quality assessment
- Meeting recording analysis with speaker labels

## Prerequisites

```bash
pip install "nemo_toolkit[audio,asr]"
pip install soundfile
```

## Usage

### Enhance Noisy Audio

```bash
python agents/audio-agent/enhance.py \
  --input noisy_recording.wav \
  --output clean_recording.wav
```

### Analyze a Meeting Recording

Transcribes audio with speaker labels:

```bash
python agents/audio-agent/analyze_meeting.py \
  --audio meeting.wav \
  --output report.json \
  --num-speakers 3
```

### Check Audio Quality

```bash
python agents/audio-agent/analyze_meeting.py \
  --audio recording.wav \
  --output report.json \
  --quality-check
```

## Output Format

`analyze_meeting.py` produces a JSON report:

```json
{
  "audio_filepath": "meeting.wav",
  "duration_seconds": 3600,
  "num_speakers_detected": 3,
  "transcript": "SPEAKER_1: Hello everyone...",
  "segments": [
    {
      "speaker": "SPEAKER_1",
      "start": 0.5,
      "end": 3.2,
      "text": "Hello everyone, let's get started."
    }
  ]
}
```

## Prompts

See [docs/PROMPTS.md](../../docs/PROMPTS.md#analysis--summarization-prompts) for meeting analysis prompt templates.
