# Audio Agent

An agent skill for audio processing and analysis using NVIDIA NeMo models.

## Features

- Transcription of meeting recordings using NeMo ASR models
- Audio quality assessment (RMS amplitude, peak, SNR estimate)
- Speaker count specification for future diarization workflows
- Meeting recording analysis with structured JSON reports

## Prerequisites

```bash
pip install "nemo_toolkit[audio,asr]"
pip install soundfile
```

## Usage

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
  "filename": "meeting.wav",
  "transcript": "Hello everyone let us get started...",
  "word_count": 42,
  "num_speakers_requested": 3,
  "notes": "Speaker diarization requires additional NeMo setup. See docs/USECASES.md for full diarization workflow.",
  "audio_quality": {
    "duration_seconds": 3600.0,
    "sample_rate": 16000,
    "rms_amplitude": 0.032451,
    "peak_amplitude": 0.981234,
    "snr_estimate_db": 25.3,
    "channels": 1
  }
}
```

## Prompts

See [docs/PROMPTS.md](../../docs/PROMPTS.md#analysis--summarization-prompts) for meeting analysis prompt templates.
