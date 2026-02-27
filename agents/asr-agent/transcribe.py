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
ASR Agent: Transcribe audio files using NVIDIA NeMo models.

Usage:
    python transcribe.py --audio path/to/audio.wav
    python transcribe.py --audio audio.wav --model nvidia/canary-1b --source-lang es --target-lang en
    python transcribe.py --audio audio.wav --output-format srt --output subtitles.srt
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using NeMo ASR models"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to input audio file (WAV, 16kHz recommended)"
    )
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-ctc-1.1b",
        help="NeMo model name or path (default: nvidia/parakeet-ctc-1.1b)",
    )
    parser.add_argument(
        "--output", default=None, help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "srt"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help="Source language code for Canary model (e.g., en, es, fr, de)",
    )
    parser.add_argument(
        "--target-lang",
        default=None,
        help="Target language for translation with Canary (e.g., en)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include word-level timestamps in output",
    )
    return parser.parse_args()


def load_asr_model(model_name: str):
    """Load the appropriate NeMo ASR model based on the model name."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print(
            "Error: NeMo ASR is not installed. Run: pip install 'nemo_toolkit[asr]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model: {model_name}", file=sys.stderr)

    if "canary" in model_name.lower():
        model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name)
    elif "rnnt" in model_name.lower():
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    elif "tdt" in model_name.lower():
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    else:
        # Default: CTC model
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)

    model.eval()
    return model


def transcribe(model, audio_path: str, timestamps: bool = False, **kwargs):
    """Run transcription and return hypothesis objects."""
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    hypotheses = model.transcribe(
        [audio_path],
        batch_size=1,
        return_hypotheses=True,
    )
    return hypotheses[0]


def format_as_srt(text: str, duration_sec: float = None) -> str:
    """Format transcript as a single SRT subtitle block."""
    duration = int(duration_sec) if duration_sec else 60
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = f"00:00:00,000 --> {hours:02d}:{minutes:02d}:{seconds:02d},000"
    return f"1\n{timestamp}\n{text}\n"


def format_output(hypothesis, output_format: str, audio_path: str) -> str:
    """Format the hypothesis into the desired output format."""
    text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)

    if output_format == "text":
        return text

    if output_format == "json":
        result = {
            "audio_filepath": audio_path,
            "transcript": text,
        }
        if hasattr(hypothesis, "timestep") and hypothesis.timestep:
            result["word_timestamps"] = hypothesis.timestep
        return json.dumps(result, indent=2, ensure_ascii=False)

    if output_format == "srt":
        return format_as_srt(text)

    return text


def main():
    args = parse_args()

    model = load_asr_model(args.model)

    # Configure Canary for multilingual/translation tasks
    if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
        if "canary" in args.model.lower():
            decode_cfg = model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            model.change_decoding_strategy(decode_cfg)

    print(f"Transcribing: {args.audio}", file=sys.stderr)
    hypothesis = transcribe(model, args.audio, timestamps=args.timestamps)

    output_text = format_output(hypothesis, args.output_format, args.audio)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Transcript saved to: {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
