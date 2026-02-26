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
Unit tests for the TTS Agent synthesize.py and batch_synthesize.py scripts.

These tests validate argument parsing, text reading, synthesis helpers, and
main workflow without requiring GPU or pretrained model downloads.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

TTS_AGENT_DIR = str(Path(__file__).resolve().parents[1])
if TTS_AGENT_DIR not in sys.path:
    sys.path.insert(0, TTS_AGENT_DIR)

import synthesize
import batch_synthesize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text_file(tmp_path):
    """Create a text file with sample content."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world, this is a test.")
    return str(text_file)


@pytest.fixture
def sentences_file(tmp_path):
    """Create a multi-line sentences file for batch synthesis."""
    sentences = tmp_path / "sentences.txt"
    sentences.write_text("First sentence.\nSecond sentence.\nThird sentence.\n")
    return str(sentences)


@pytest.fixture
def empty_text_file(tmp_path):
    """Create an empty text file."""
    text_file = tmp_path / "empty.txt"
    text_file.write_text("")
    return str(text_file)


@pytest.fixture
def mock_audio_array():
    """Create a mock audio numpy array."""
    return np.random.randn(22050).astype(np.float32)  # 1 second of audio


# ===========================================================================
# Tests for synthesize.py
# ===========================================================================


class TestSynthesizeParseArgs:
    """Test argument parsing for synthesize.py."""

    @pytest.mark.unit
    def test_text_input(self):
        """--text and --output are accepted."""
        with patch(
            "sys.argv",
            ["synthesize.py", "--text", "Hello", "--output", "out.wav"],
        ):
            args = synthesize.parse_args()
        assert args.text == "Hello"
        assert args.output == "out.wav"
        assert args.input_file is None

    @pytest.mark.unit
    def test_file_input(self):
        """--input-file and --output are accepted."""
        with patch(
            "sys.argv",
            ["synthesize.py", "--input-file", "article.txt", "--output", "out.wav"],
        ):
            args = synthesize.parse_args()
        assert args.input_file == "article.txt"
        assert args.text is None

    @pytest.mark.unit
    def test_text_and_file_mutually_exclusive(self):
        """--text and --input-file cannot both be specified."""
        with pytest.raises(SystemExit):
            with patch(
                "sys.argv",
                ["synthesize.py", "--text", "Hi", "--input-file", "f.txt", "--output", "o.wav"],
            ):
                synthesize.parse_args()

    @pytest.mark.unit
    def test_requires_input(self):
        """Either --text or --input-file is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["synthesize.py", "--output", "out.wav"]):
                synthesize.parse_args()

    @pytest.mark.unit
    def test_requires_output(self):
        """--output is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["synthesize.py", "--text", "Hello"]):
                synthesize.parse_args()

    @pytest.mark.unit
    def test_default_models(self):
        """Default acoustic model and vocoder are set correctly."""
        with patch(
            "sys.argv",
            ["synthesize.py", "--text", "Hello", "--output", "out.wav"],
        ):
            args = synthesize.parse_args()
        assert args.acoustic_model == "nvidia/tts_en_fastpitch"
        assert args.vocoder == "nvidia/tts_hifigan"
        assert args.sample_rate == 22050

    @pytest.mark.unit
    def test_custom_sample_rate(self):
        """Custom sample rate is parsed."""
        with patch(
            "sys.argv",
            ["synthesize.py", "--text", "Hi", "--output", "o.wav", "--sample-rate", "44100"],
        ):
            args = synthesize.parse_args()
        assert args.sample_rate == 44100


class TestReadTextFile:
    """Test text file reading."""

    @pytest.mark.unit
    def test_read_existing_file(self, sample_text_file):
        """Should read and strip content from file."""
        text = synthesize.read_text_file(sample_text_file)
        assert text == "Hello world, this is a test."

    @pytest.mark.unit
    def test_read_nonexistent_file(self):
        """Should exit when file doesn't exist."""
        with pytest.raises(SystemExit):
            synthesize.read_text_file("/nonexistent/file.txt")

    @pytest.mark.unit
    def test_read_empty_file(self, empty_text_file):
        """Should return empty string for empty file."""
        text = synthesize.read_text_file(empty_text_file)
        assert text == ""


class TestSynthesizeSpeech:
    """Test speech synthesis function."""

    @pytest.mark.unit
    def test_synthesize_returns_numpy(self):
        """synthesize_speech returns a 1D numpy array."""
        # Create mock torch module and tensor behavior
        mock_audio_tensor = MagicMock()
        mock_audio_tensor.to.return_value = mock_audio_tensor
        mock_audio_tensor.detach.return_value = mock_audio_tensor
        mock_audio_tensor.numpy.return_value = np.random.randn(1, 22050).astype(np.float32)

        spec_gen = MagicMock()
        vocoder = MagicMock()

        spec_gen.parse.return_value = MagicMock()
        spec_gen.generate_spectrogram.return_value = MagicMock()
        vocoder.convert_spectrogram_to_audio.return_value = mock_audio_tensor

        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = synthesize.synthesize_speech(spec_gen, vocoder, "Hello")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1


class TestSynthesizeMain:
    """Test main() workflow for synthesize.py."""

    @pytest.mark.unit
    def test_main_empty_text_exits(self, empty_text_file, tmp_path):
        """main() exits when input text is empty."""
        output = str(tmp_path / "speech.wav")

        with patch(
            "sys.argv",
            ["synthesize.py", "--input-file", empty_text_file, "--output", output],
        ):
            with pytest.raises(SystemExit):
                synthesize.main()


# ===========================================================================
# Tests for batch_synthesize.py
# ===========================================================================


class TestBatchSynthesizeParseArgs:
    """Test argument parsing for batch_synthesize.py."""

    @pytest.mark.unit
    def test_required_args(self):
        """--input and --output-dir are required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["batch_synthesize.py"]):
                batch_synthesize.parse_args()

    @pytest.mark.unit
    def test_defaults(self):
        """Default values are set correctly."""
        with patch(
            "sys.argv",
            ["batch_synthesize.py", "--input", "s.txt", "--output-dir", "out/"],
        ):
            args = batch_synthesize.parse_args()
        assert args.prefix == "output"
        assert args.acoustic_model == "nvidia/tts_en_fastpitch"
        assert args.vocoder == "nvidia/tts_hifigan"
        assert args.sample_rate == 22050


class TestBatchSynthesizeMain:
    """Test batch_synthesize main() workflow."""

    @pytest.mark.unit
    def test_nonexistent_input_file(self, tmp_path):
        """main() exits when input file doesn't exist."""
        with patch(
            "sys.argv",
            [
                "batch_synthesize.py",
                "--input", "/nonexistent/sentences.txt",
                "--output-dir", str(tmp_path),
            ],
        ):
            with pytest.raises(SystemExit):
                batch_synthesize.main()

    @pytest.mark.unit
    def test_empty_input_file(self, empty_text_file, tmp_path):
        """main() exits when input file has no sentences."""
        with patch(
            "sys.argv",
            [
                "batch_synthesize.py",
                "--input", empty_text_file,
                "--output-dir", str(tmp_path),
            ],
        ):
            with pytest.raises(SystemExit):
                batch_synthesize.main()
