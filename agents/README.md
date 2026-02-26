# NeMo Agent Skills

AI agent skills for automating NVIDIA NeMo speech AI workflows.

## Available Agents

| Agent | Description | Use Cases |
|-------|-------------|-----------|
| [ASR Agent](asr-agent/) | Automatic Speech Recognition automation | Transcription, batch processing, diarization |
| [TTS Agent](tts-agent/) | Text-to-Speech synthesis | Audio generation, voice cloning, batch narration |
| [Audio Agent](audio-agent/) | Audio processing and analysis | Enhancement, speaker ID, call analytics |

## Quick Start

Each agent is a self-contained Python script that can be run from the command line:

```bash
# ASR: Transcribe an audio file
python agents/asr-agent/transcribe.py --audio path/to/audio.wav

# ASR: Batch transcription
python agents/asr-agent/batch_transcribe.py --input-dir audio/ --output transcripts.json

# TTS: Generate speech
python agents/tts-agent/synthesize.py --text "Hello, world!" --output speech.wav

# TTS: Batch generation from file
python agents/tts-agent/batch_synthesize.py --input sentences.txt --output-dir audio/

# Audio: Enhance noisy audio
python agents/audio-agent/enhance.py --input noisy.wav --output clean.wav

# Audio: Analyze a meeting recording
python agents/audio-agent/analyze_meeting.py --audio meeting.wav --output report.json
```

## Prerequisites

Install NeMo with the relevant collection before using agents:

```bash
pip install "nemo_toolkit[asr]"    # For ASR agent
pip install "nemo_toolkit[tts]"    # For TTS agent
pip install "nemo_toolkit[audio]"  # For Audio agent
pip install "nemo_toolkit[all]"    # For all agents
```

## Agent Architecture

Each agent follows this pattern:

```
agents/
└── <agent-name>/
    ├── README.md          # Agent documentation
    ├── <main_script>.py   # Main entry point
    ├── requirements.txt   # Agent-specific dependencies
    └── examples/          # Example usage scripts
```

## Documentation

For detailed prompt templates and workflow guidance, see [docs/PROMPTS.md](../docs/PROMPTS.md).
