#!/usr/bin/env python3
"""
ASR Agent: Batch transcribe audio files from a directory or manifest.

Usage:
    python batch_transcribe.py --input-dir audio/ --output transcripts.json
    python batch_transcribe.py --manifest data/manifest.json --output results.json
    python batch_transcribe.py --input-dir audio/ --output results.json --batch-size 16
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using NeMo ASR models"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir",
        help="Directory containing audio files",
    )
    group.add_argument(
        "--manifest",
        help="NeMo manifest JSON file with audio paths",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for transcripts",
    )
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-ctc-1.1b",
        help="NeMo model name (default: nvidia/parakeet-ctc-1.1b)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for transcription (default: 8)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directory for audio files",
    )
    return parser.parse_args()


def collect_audio_files(input_dir: str, recursive: bool = False):
    """Collect all supported audio files from a directory."""
    audio_files = []
    pattern = "**/*" if recursive else "*"
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        matches = glob.glob(
            os.path.join(input_dir, pattern + ext), recursive=recursive
        )
        audio_files.extend(matches)
    return sorted(audio_files)


def load_manifest(manifest_path: str):
    """Load audio files from a NeMo manifest."""
    audio_files = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                audio_files.append(data["audio_filepath"])
    return audio_files


def load_model(model_name: str):
    """Load NeMo ASR model."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print(
            "Error: NeMo ASR is not installed. Run: pip install 'nemo_toolkit[asr]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model: {model_name}", file=sys.stderr)
    if "rnnt" in model_name.lower() or "tdt" in model_name.lower():
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    elif "canary" in model_name.lower():
        model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
    model.eval()
    return model


def batch_transcribe(model, audio_files, batch_size: int = 8):
    """Transcribe a list of audio files in batches."""
    results = []
    total = len(audio_files)

    for i in range(0, total, batch_size):
        batch = audio_files[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)...",
            file=sys.stderr,
        )

        # Filter out files that don't exist
        valid_batch = [f for f in batch if os.path.exists(f)]
        if len(valid_batch) < len(batch):
            missing = set(batch) - set(valid_batch)
            for f in missing:
                print(f"  Warning: File not found, skipping: {f}", file=sys.stderr)

        if not valid_batch:
            continue

        hypotheses = model.transcribe(
            valid_batch,
            batch_size=len(valid_batch),
            return_hypotheses=True,
        )

        for audio_path, hypothesis in zip(valid_batch, hypotheses):
            text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
            results.append({
                "audio_filepath": audio_path,
                "filename": Path(audio_path).name,
                "transcript": text,
            })

    return results


def main():
    args = parse_args()

    # Collect audio files
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Directory not found: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        audio_files = collect_audio_files(args.input_dir, args.recursive)
        if not audio_files:
            print(f"No audio files found in: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(audio_files)} audio files", file=sys.stderr)
    else:
        audio_files = load_manifest(args.manifest)
        print(f"Loaded {len(audio_files)} files from manifest", file=sys.stderr)

    # Load model and transcribe
    model = load_model(args.model)
    results = batch_transcribe(model, audio_files, batch_size=args.batch_size)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"\nCompleted: {len(results)}/{len(audio_files)} files transcribed",
        file=sys.stderr,
    )
    print(f"Output saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
