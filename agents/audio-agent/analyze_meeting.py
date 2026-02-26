#!/usr/bin/env python3
"""
Audio Agent: Analyze meeting recordings with ASR transcription and speaker labels.

Usage:
    python analyze_meeting.py --audio meeting.wav --output report.json
    python analyze_meeting.py --audio meeting.wav --output report.json --num-speakers 3
    python analyze_meeting.py --audio meeting.wav --output report.json --quality-check
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze meeting audio with transcription and speaker information"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to audio file (WAV, 16kHz)"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON report file path"
    )
    parser.add_argument(
        "--asr-model",
        default="nvidia/parakeet-ctc-1.1b",
        help="NeMo ASR model for transcription",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (auto-detect if not specified)",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Include audio quality metrics in report",
    )
    return parser.parse_args()


def check_audio_quality(audio_path: str) -> dict:
    """Compute basic audio quality metrics."""
    try:
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Mix to mono

        duration = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        snr_estimate = 20 * np.log10(rms / (1e-10 + np.std(audio - np.mean(audio))))

        return {
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "rms_amplitude": round(rms, 6),
            "peak_amplitude": round(peak, 6),
            "snr_estimate_db": round(float(snr_estimate), 2),
            "channels": 1 if audio.ndim == 1 else audio.shape[1],
        }
    except ImportError:
        print("Warning: soundfile not installed, skipping quality check.", file=sys.stderr)
        return {}


def transcribe_audio(audio_path: str, model_name: str) -> str:
    """Transcribe audio using NeMo ASR."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print(
            "Error: NeMo ASR not installed. Run: pip install 'nemo_toolkit[asr]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading ASR model: {model_name}", file=sys.stderr)
    if "rnnt" in model_name.lower() or "tdt" in model_name.lower():
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    elif "canary" in model_name.lower():
        model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
    model.eval()

    print("Transcribing audio...", file=sys.stderr)
    hypotheses = model.transcribe([audio_path], batch_size=1, return_hypotheses=True)
    hypothesis = hypotheses[0]
    text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
    return text


def build_report(
    audio_path: str,
    transcript: str,
    quality_metrics: dict,
    num_speakers: int,
) -> dict:
    """Build the analysis report."""
    report = {
        "audio_filepath": audio_path,
        "filename": Path(audio_path).name,
        "transcript": transcript,
        "word_count": len(transcript.split()),
        "num_speakers_requested": num_speakers,
        "notes": (
            "Speaker diarization requires additional NeMo setup. "
            "See docs/USECASES.md for full diarization workflow."
        ),
    }

    if quality_metrics:
        report["audio_quality"] = quality_metrics

    return report


def main():
    args = parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Quality check
    quality_metrics = {}
    if args.quality_check:
        print("Running audio quality check...", file=sys.stderr)
        quality_metrics = check_audio_quality(args.audio)
        if quality_metrics:
            print(
                f"  Duration: {quality_metrics.get('duration_seconds', 'N/A')}s, "
                f"Sample rate: {quality_metrics.get('sample_rate', 'N/A')}Hz",
                file=sys.stderr,
            )

    # Transcription
    transcript = transcribe_audio(args.audio, args.asr_model)
    print(f"\nTranscript preview: {transcript[:200]}...", file=sys.stderr)

    # Build report
    report = build_report(
        args.audio,
        transcript,
        quality_metrics,
        args.num_speakers,
    )

    # Save report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
