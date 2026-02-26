#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TTS Agent: Convert text to speech using NVIDIA NeMo models.

Usage:
    python synthesize.py --text "Hello, world!" --output speech.wav
    python synthesize.py --input-file article.txt --output narration.wav
    python synthesize.py --text "Hello!" --output speech.wav --sample-rate 22050
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize speech from text using NeMo TTS models"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to synthesize")
    input_group.add_argument("--input-file", help="Path to text file to synthesize")
    parser.add_argument(
        "--output", required=True, help="Output WAV file path"
    )
    parser.add_argument(
        "--acoustic-model",
        default="nvidia/tts_en_fastpitch",
        help="Acoustic model name (default: nvidia/tts_en_fastpitch)",
    )
    parser.add_argument(
        "--vocoder",
        default="nvidia/tts_hifigan",
        help="Vocoder model name (default: nvidia/tts_hifigan)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Output sample rate in Hz (default: 22050)",
    )
    return parser.parse_args()


def load_tts_models(acoustic_model_name: str, vocoder_name: str):
    """Load NeMo TTS acoustic model and vocoder."""
    try:
        import nemo.collections.tts as nemo_tts
    except ImportError:
        print(
            "Error: NeMo TTS is not installed. Run: pip install 'nemo_toolkit[tts]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading acoustic model: {acoustic_model_name}", file=sys.stderr)
    spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(acoustic_model_name)
    spec_generator.eval()

    print(f"Loading vocoder: {vocoder_name}", file=sys.stderr)
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(vocoder_name)
    vocoder.eval()

    return spec_generator, vocoder


def synthesize_speech(spec_generator, vocoder, text: str):
    """Generate audio numpy array from text."""
    import torch

    with torch.no_grad():
        parsed = spec_generator.parse(text)
        spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    return audio.to("cpu").detach().numpy().reshape(-1)


def read_text_file(file_path: str) -> str:
    """Read text from a file."""
    if not os.path.exists(file_path):
        print(f"Error: Input file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()


def main():
    args = parse_args()

    # Get input text
    if args.text:
        text = args.text
    else:
        text = read_text_file(args.input_file)
        print(
            f"Loaded text from {args.input_file} ({len(text)} characters)",
            file=sys.stderr,
        )

    if not text:
        print("Error: Input text is empty.", file=sys.stderr)
        sys.exit(1)

    # Load models
    spec_generator, vocoder = load_tts_models(args.acoustic_model, args.vocoder)

    # Synthesize
    print(f"Synthesizing speech...", file=sys.stderr)
    audio_array = synthesize_speech(spec_generator, vocoder, text)

    # Save output
    try:
        import soundfile as sf
    except ImportError:
        print(
            "Error: soundfile is not installed. Run: pip install soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, audio_array, args.sample_rate)

    duration = len(audio_array) / args.sample_rate
    print(
        f"Generated {duration:.1f}s of audio, saved to: {args.output}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
