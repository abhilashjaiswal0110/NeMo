# NeMo Use Cases Guide

Practical, code-ready examples for common NeMo use cases across ASR, TTS, and Audio Processing.

## Table of Contents

- [Automatic Speech Recognition (ASR)](#automatic-speech-recognition-asr)
  - [Transcribe an Audio File](#transcribe-an-audio-file)
  - [Batch Transcription](#batch-transcription)
  - [Multilingual Transcription with Canary](#multilingual-transcription-with-canary)
  - [Real-time Streaming ASR](#real-time-streaming-asr)
  - [Fine-tune ASR on Custom Data](#fine-tune-asr-on-custom-data)
- [Text to Speech (TTS)](#text-to-speech-tts)
  - [Generate Speech from Text](#generate-speech-from-text)
  - [Batch TTS Generation](#batch-tts-generation)
  - [Fine-tune TTS on a Custom Voice](#fine-tune-tts-on-a-custom-voice)
- [Audio Processing](#audio-processing)
  - [Speech Enhancement / Noise Reduction](#speech-enhancement--noise-reduction)
  - [Speaker Diarization](#speaker-diarization)
- [End-to-End Pipelines](#end-to-end-pipelines)
  - [Audio → Transcript → Summary](#audio--transcript--summary)
  - [Text → Speech → Audio File](#text--speech--audio-file)

---

## Automatic Speech Recognition (ASR)

### Transcribe an Audio File

The fastest way to transcribe speech using a pre-trained Parakeet model:

```python
import nemo.collections.asr as nemo_asr

# Load the Parakeet CTC model (downloads automatically on first run)
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/parakeet-ctc-1.1b"
)

# Transcribe a single audio file (WAV, 16kHz mono recommended)
transcript = asr_model.transcribe(["path/to/audio.wav"])
print(transcript[0])
```

**Supported models** (from NGC / HuggingFace):

| Model | Size | Best For |
|-------|------|----------|
| `nvidia/parakeet-ctc-1.1b` | 1.1B | High accuracy English ASR |
| `nvidia/parakeet-rnnt-1.1b` | 1.1B | Streaming + accuracy tradeoff |
| `nvidia/parakeet-tdt-1.1b` | 1.1B | Speed + accuracy (TDT) |
| `nvidia/canary-1b` | 1B | Multilingual + translation |

### Batch Transcription

Efficiently transcribe many audio files:

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/parakeet-ctc-1.1b"
)

audio_files = [
    "audio/meeting_01.wav",
    "audio/meeting_02.wav",
    "audio/interview.wav",
]

# Batch transcription with timestamps
transcripts = asr_model.transcribe(
    audio_files,
    batch_size=8,        # adjust to GPU memory
    return_hypotheses=True,
)

for audio_path, hypothesis in zip(audio_files, transcripts):
    print(f"{audio_path}: {hypothesis.text}")
    if hypothesis.timestep:
        print(f"  Word timestamps: {hypothesis.timestep}")
```

### Multilingual Transcription with Canary

Canary supports English, Spanish, French, and German with translation:

```python
import nemo.collections.asr as nemo_asr

canary_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
    "nvidia/canary-1b"
)

# Update decode params for multilingual transcription
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)

# Transcribe Spanish audio
transcripts = canary_model.transcribe(
    ["spanish_audio.wav"],
    batch_size=1,
    # Canary auto-detects language; you can also specify:
    # pnc="yes",         # Punctuation & Capitalization
    # source_lang="es",  # source language code
    # target_lang="en",  # translate to English
)
print(transcripts[0])
```

### Real-time Streaming ASR

For low-latency applications using chunked audio:

```python
import nemo.collections.asr as nemo_asr
import numpy as np

# Use a streaming-capable model (RNNT or TDT)
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="nvidia/parakeet-rnnt-1.1b"
)

# Configure streaming parameters
asr_model.eval()

# Process audio in chunks (e.g., from a microphone)
chunk_size = 16000  # 1 second at 16kHz

def process_audio_stream(audio_chunks):
    """Process audio stream chunk by chunk."""
    buffer = []
    for chunk in audio_chunks:
        buffer.append(chunk)
        if len(buffer) * chunk_size >= 32000:  # 2-second window
            audio_array = np.concatenate(buffer)
            # Write to temp file and transcribe
            import soundfile as sf
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_array, 16000)
                transcript = asr_model.transcribe([f.name])
                os.unlink(f.name)
            yield transcript[0]
            buffer = []
```

### Fine-tune ASR on Custom Data

Fine-tune Parakeet on your domain-specific data:

```python
# 1. Prepare manifest file (JSON Lines format)
# Each line: {"audio_filepath": "path.wav", "duration": 5.2, "text": "transcription"}

# 2. Create fine-tuning config
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from pytorch_lightning import Trainer

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/parakeet-ctc-1.1b"
)

# Update data configs
model.cfg.train_ds.manifest_filepath = "data/train_manifest.json"
model.cfg.validation_ds.manifest_filepath = "data/val_manifest.json"
model.cfg.train_ds.batch_size = 16
model.cfg.optim.lr = 1e-5  # Lower LR for fine-tuning

trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    log_every_n_steps=10,
)

model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

trainer.fit(model)

# Save the fine-tuned model
model.save_to("finetuned_asr_model.nemo")
```

---

## Text to Speech (TTS)

### Generate Speech from Text

```python
import nemo.collections.tts as nemo_tts
import soundfile as sf

# Load FastPitch (acoustic model) and HiFi-GAN (vocoder)
spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
    "nvidia/tts_en_fastpitch"
)
vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
    "nvidia/tts_hifigan"
)

# Generate speech
text = "Hello! Welcome to NVIDIA NeMo text to speech synthesis."
parsed = spec_generator.parse(text)
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save to file
audio_numpy = audio.to("cpu").detach().numpy().reshape(-1)
sf.write("output_speech.wav", audio_numpy, 22050)
print("Saved to output_speech.wav")
```

### Batch TTS Generation

```python
import nemo.collections.tts as nemo_tts
import soundfile as sf
import os

spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
    "nvidia/tts_en_fastpitch"
)
vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
    "nvidia/tts_hifigan"
)

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "NVIDIA NeMo is a powerful speech AI framework.",
    "Neural networks have revolutionized speech synthesis.",
]

os.makedirs("tts_output", exist_ok=True)

for i, sentence in enumerate(sentences):
    parsed = spec_generator.parse(sentence)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    audio_numpy = audio.to("cpu").detach().numpy().reshape(-1)
    output_path = f"tts_output/sentence_{i+1:03d}.wav"
    sf.write(output_path, audio_numpy, 22050)
    print(f"Generated: {output_path}")
```

### Fine-tune TTS on a Custom Voice

```python
# Requires ~30+ minutes of high-quality recordings at 22kHz
from pytorch_lightning import Trainer
import nemo.collections.tts as nemo_tts

# Load base FastPitch model
model = nemo_tts.models.FastPitchModel.from_pretrained(
    "nvidia/tts_en_fastpitch"
)

# Configure with your voice data
model.cfg.train_ds.dataset.manifest_filepath = "data/custom_voice_train.json"
model.cfg.validation_ds.dataset.manifest_filepath = "data/custom_voice_val.json"
model.cfg.optim.lr = 1e-4

trainer = Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)

trainer.fit(model)
model.save_to("custom_voice.nemo")
```

---

## Audio Processing

### Speech Enhancement / Noise Reduction

```python
import nemo.collections.audio as nemo_audio
import soundfile as sf

# Load a speech enhancement model
enhancement_model = nemo_audio.models.EncMaskDecAudioToAudioModel.from_pretrained(
    "nvidia/nemo-audio-en-speech-enhancement"
)

# Process a noisy audio file
noisy_audio = "noisy_recording.wav"
enhanced_audio = enhancement_model.process(noisy_audio)

# Save the enhanced audio
sf.write("enhanced_output.wav", enhanced_audio, samplerate=16000)
```

### Speaker Diarization

Identify who spoke when in a multi-speaker recording:

```python
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf

# Configure diarization
diarizer_cfg = OmegaConf.create({
    "diarizer": {
        "manifest_filepath": "input_manifest.json",
        "out_dir": "diarization_output",
        "oracle_vad": False,
        "collar": 0.25,
        "ignore_overlap": True,
        "vad": {
            "model_path": "vad_multilingual_marblenet",
            "external_vad_manifest": None,
            "parameters": {
                "window_length_in_sec": 0.15,
                "shift_length_in_sec": 0.01,
            },
        },
        "speaker_embeddings": {
            "model_path": "titanet_large",
            "parameters": {
                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                "multiscale_weights": [1, 1, 1, 1, 1],
            },
        },
        "clustering": {
            "parameters": {
                "oracle_num_speakers": False,
                "max_num_speakers": 8,
            }
        },
        "msdd_model": {
            "model_path": "diar_msdd_telephony",
            "parameters": {
                "sigmoid_threshold": [0.7],
            },
        },
    }
})

# Create input manifest
import json
manifest_data = {
    "audio_filepath": "meeting_recording.wav",
    "offset": 0,
    "duration": None,
    "label": "infer",
    "text": "-",
    "num_speakers": None,
}

with open("input_manifest.json", "w") as f:
    json.dump(manifest_data, f)
    f.write("\n")

# Run diarization
diarizer = NeuralDiarizer(cfg=diarizer_cfg)
diarizer.diarize()
# Output: RTTM files in diarization_output/
```

---

## End-to-End Pipelines

### Audio → Transcript → Summary

```python
import nemo.collections.asr as nemo_asr

# Step 1: Transcribe audio
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    "nvidia/parakeet-ctc-1.1b"
)
transcript = asr_model.transcribe(["meeting.wav"])[0]
print(f"Transcript:\n{transcript}\n")

# Step 2: (Optional) Use an LLM to summarize
# This example shows using the transcript with any LLM API
summary_prompt = f"""
Summarize the following meeting transcript in bullet points:

{transcript}

Summary:
"""
# Pass summary_prompt to your preferred LLM (OpenAI, Claude, etc.)
print(f"Prompt ready for summarization:\n{summary_prompt[:200]}...")
```

### Text → Speech → Audio File

```python
import nemo.collections.tts as nemo_tts
import soundfile as sf

spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
    "nvidia/tts_en_fastpitch"
)
vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
    "nvidia/tts_hifigan"
)

def text_to_speech(text: str, output_path: str, sample_rate: int = 22050):
    """Convert text to speech and save to file."""
    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    audio_numpy = audio.to("cpu").detach().numpy().reshape(-1)
    sf.write(output_path, audio_numpy, sample_rate)
    return output_path

# Example usage
article_text = """
NVIDIA NeMo is a framework for building AI models.
It supports speech recognition, text to speech, and audio processing.
"""
text_to_speech(article_text, "article_narration.wav")
```

---

## Next Steps

- 🎯 See [PROMPTS.md](PROMPTS.md) for prompt templates used with NeMo
- 🤖 Use the [Agents](../agents/README.md) for automated NeMo workflows
- 📓 Explore [tutorials/](../tutorials/) for Jupyter notebook walkthroughs
- 🔧 See [LOCAL_SETUP.md](LOCAL_SETUP.md) for installation help
