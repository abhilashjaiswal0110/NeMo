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
Unit tests for the Audio Agent analyze_meeting.py script.

These tests validate argument parsing, audio quality checking, report building,
and main workflow without requiring GPU or pretrained model downloads.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

AUDIO_AGENT_DIR = str(Path(__file__).resolve().parents[1])
if AUDIO_AGENT_DIR not in sys.path:
    sys.path.insert(0, AUDIO_AGENT_DIR)

import analyze_meeting


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_wav_file(tmp_path):
    """Create a minimal valid WAV file for testing."""
    import struct

    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    audio_data = np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate).astype(np.float32)

    wav_file = tmp_path / "test_meeting.wav"

    # Write a minimal WAV file manually
    import wave
    import array

    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(str(wav_file), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return str(wav_file)


# ===========================================================================
# Tests for analyze_meeting.py
# ===========================================================================


class TestAnalyzeMeetingParseArgs:
    """Test argument parsing for analyze_meeting.py."""

    @pytest.mark.unit
    def test_required_args(self):
        """--audio and --output are required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["analyze_meeting.py"]):
                analyze_meeting.parse_args()

    @pytest.mark.unit
    def test_requires_audio(self):
        """--audio is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["analyze_meeting.py", "--output", "report.json"]):
                analyze_meeting.parse_args()

    @pytest.mark.unit
    def test_requires_output(self):
        """--output is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["analyze_meeting.py", "--audio", "meeting.wav"]):
                analyze_meeting.parse_args()

    @pytest.mark.unit
    def test_all_args(self):
        """All arguments are parsed correctly."""
        with patch(
            "sys.argv",
            [
                "analyze_meeting.py",
                "--audio", "meeting.wav",
                "--output", "report.json",
                "--asr-model", "nvidia/canary-1b",
                "--num-speakers", "3",
                "--quality-check",
            ],
        ):
            args = analyze_meeting.parse_args()
        assert args.audio == "meeting.wav"
        assert args.output == "report.json"
        assert args.asr_model == "nvidia/canary-1b"
        assert args.num_speakers == 3
        assert args.quality_check is True

    @pytest.mark.unit
    def test_default_values(self):
        """Default values are correct."""
        with patch(
            "sys.argv",
            ["analyze_meeting.py", "--audio", "m.wav", "--output", "r.json"],
        ):
            args = analyze_meeting.parse_args()
        assert args.asr_model == "nvidia/parakeet-ctc-1.1b"
        assert args.num_speakers is None
        assert args.quality_check is False


class TestCheckAudioQuality:
    """Test audio quality check function."""

    @pytest.mark.unit
    def test_quality_metrics_structure(self, tmp_wav_file):
        """Quality metrics should contain expected keys."""
        metrics = analyze_meeting.check_audio_quality(tmp_wav_file)
        assert "duration_seconds" in metrics
        assert "sample_rate" in metrics
        assert "rms_amplitude" in metrics
        assert "peak_amplitude" in metrics
        assert "snr_estimate_db" in metrics
        assert "channels" in metrics

    @pytest.mark.unit
    def test_quality_metrics_types(self, tmp_wav_file):
        """Quality metrics should have correct types."""
        metrics = analyze_meeting.check_audio_quality(tmp_wav_file)
        assert isinstance(metrics["duration_seconds"], float)
        assert isinstance(metrics["sample_rate"], int)
        assert isinstance(metrics["rms_amplitude"], float)
        assert isinstance(metrics["peak_amplitude"], float)
        assert isinstance(metrics["snr_estimate_db"], float)

    @pytest.mark.unit
    def test_quality_values_reasonable(self, tmp_wav_file):
        """Quality values should be in reasonable ranges."""
        metrics = analyze_meeting.check_audio_quality(tmp_wav_file)
        assert metrics["duration_seconds"] > 0
        assert metrics["sample_rate"] == 16000
        assert 0.0 <= metrics["rms_amplitude"] <= 1.0
        assert 0.0 <= metrics["peak_amplitude"] <= 1.0

    @pytest.mark.unit
    def test_quality_check_missing_soundfile(self):
        """Should return empty dict when soundfile not available."""
        with patch.dict("sys.modules", {"soundfile": None}):
            with patch("builtins.__import__", side_effect=ImportError("No soundfile")):
                # The function catches ImportError internally
                metrics = analyze_meeting.check_audio_quality("/tmp/dummy.wav")
                assert metrics == {}


class TestBuildReport:
    """Test report building function."""

    @pytest.mark.unit
    def test_basic_report_structure(self):
        """Report should contain required fields."""
        report = analyze_meeting.build_report(
            audio_path="meeting.wav",
            transcript="Hello this is a meeting",
            quality_metrics={},
            num_speakers=2,
        )
        assert report["audio_filepath"] == "meeting.wav"
        assert report["filename"] == "meeting.wav"
        assert report["transcript"] == "Hello this is a meeting"
        assert report["word_count"] == 5
        assert report["num_speakers_requested"] == 2
        assert "notes" in report

    @pytest.mark.unit
    def test_report_with_quality_metrics(self):
        """Report should include quality metrics when provided."""
        metrics = {"duration_seconds": 60.0, "sample_rate": 16000}
        report = analyze_meeting.build_report(
            audio_path="audio.wav",
            transcript="Test",
            quality_metrics=metrics,
            num_speakers=None,
        )
        assert "audio_quality" in report
        assert report["audio_quality"]["duration_seconds"] == 60.0

    @pytest.mark.unit
    def test_report_without_quality_metrics(self):
        """Report should not include audio_quality when empty."""
        report = analyze_meeting.build_report(
            audio_path="audio.wav",
            transcript="Test",
            quality_metrics={},
            num_speakers=None,
        )
        assert "audio_quality" not in report

    @pytest.mark.unit
    def test_report_word_count_accuracy(self):
        """Word count should match actual words."""
        transcript = "one two three four five six"
        report = analyze_meeting.build_report("a.wav", transcript, {}, None)
        assert report["word_count"] == 6

    @pytest.mark.unit
    def test_report_empty_transcript(self):
        """Empty transcript should have word_count of 0."""
        report = analyze_meeting.build_report("a.wav", "", {}, None)
        # "".split() returns [] so len is 0
        assert report["word_count"] == 0


class TestAnalyzeMeetingMain:
    """Test main() workflow for analyze_meeting.py."""

    @pytest.mark.unit
    def test_main_nonexistent_audio(self, tmp_path):
        """main() exits when audio file doesn't exist."""
        output = str(tmp_path / "report.json")
        with patch(
            "sys.argv",
            ["analyze_meeting.py", "--audio", "/nonexistent/audio.wav", "--output", output],
        ):
            with pytest.raises(SystemExit):
                analyze_meeting.main()

    @pytest.mark.unit
    def test_main_produces_report(self, tmp_wav_file, tmp_path):
        """main() produces a valid JSON report."""
        output = str(tmp_path / "report.json")

        with patch(
            "sys.argv",
            ["analyze_meeting.py", "--audio", tmp_wav_file, "--output", output],
        ):
            with patch(
                "analyze_meeting.transcribe_audio",
                return_value="Hello this is the meeting transcript",
            ):
                analyze_meeting.main()

        assert os.path.exists(output)
        with open(output) as f:
            report = json.load(f)
        assert report["transcript"] == "Hello this is the meeting transcript"
        assert report["word_count"] == 6

    @pytest.mark.unit
    def test_main_with_quality_check(self, tmp_wav_file, tmp_path):
        """main() includes quality metrics when --quality-check is set."""
        output = str(tmp_path / "report.json")

        with patch(
            "sys.argv",
            [
                "analyze_meeting.py",
                "--audio", tmp_wav_file,
                "--output", output,
                "--quality-check",
            ],
        ):
            with patch(
                "analyze_meeting.transcribe_audio",
                return_value="Test transcript",
            ):
                analyze_meeting.main()

        with open(output) as f:
            report = json.load(f)
        assert "audio_quality" in report
        assert report["audio_quality"]["sample_rate"] == 16000

    @pytest.mark.unit
    def test_main_creates_output_directory(self, tmp_wav_file, tmp_path):
        """main() creates parent directories for output file."""
        output = str(tmp_path / "nested" / "dir" / "report.json")

        with patch(
            "sys.argv",
            ["analyze_meeting.py", "--audio", tmp_wav_file, "--output", output],
        ):
            with patch(
                "analyze_meeting.transcribe_audio",
                return_value="Transcript",
            ):
                analyze_meeting.main()

        assert os.path.exists(output)
