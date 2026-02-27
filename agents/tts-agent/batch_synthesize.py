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
TTS Agent: Batch synthesize speech from a text file (one sentence per line).

Usage:
    python batch_synthesize.py --input sentences.txt --output-dir audio/
    python batch_synthesize.py --input sentences.txt --output-dir audio/ --prefix narration
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch synthesize speech from text file"
    )
    parser.add_argument(
        "--input", required=True, help="Text file with one sentence per line"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save audio files"
    )
    parser.add_argument(
        "--prefix",
        default="output",
        help="Filename prefix for output files (default: output)",
    )
    parser.add_argument(
        "--acoustic-model",
        default="nvidia/tts_en_fastpitch",
        help="Acoustic model name",
    )
    parser.add_argument(
        "--vocoder",
        default="nvidia/tts_hifigan",
        help="Vocoder model name",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Output sample rate in Hz (default: 22050)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read sentences
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if not sentences:
        print("Error: No sentences found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sentences)} sentences to synthesize", file=sys.stderr)

    # Load TTS models
    try:
        import nemo.collections.tts as nemo_tts
    except ImportError:
        print(
            "Error: NeMo TTS not installed. Run: pip install 'nemo_toolkit[tts]'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import soundfile as sf
    except ImportError:
        print(
            "Error: soundfile not installed. Run: pip install soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch

    print(f"Loading models...", file=sys.stderr)
    spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
        args.acoustic_model
    )
    spec_generator.eval()
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(args.vocoder)
    vocoder.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for i, sentence in enumerate(sentences, start=1):
        output_path = os.path.join(
            args.output_dir, f"{args.prefix}_{i:04d}.wav"
        )
        print(f"[{i}/{len(sentences)}] Synthesizing: {sentence[:60]}...", file=sys.stderr)

        with torch.no_grad():
            parsed = spec_generator.parse(sentence)
            spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        audio_numpy = audio.to("cpu").detach().numpy().reshape(-1)
        sf.write(output_path, audio_numpy, args.sample_rate)

    print(
        f"\nCompleted: {len(sentences)} files saved to {args.output_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
