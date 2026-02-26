# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- `docs/LOCAL_SETUP.md` — Step-by-step local installation guide for all platforms
- `docs/USECASES.md` — Practical code examples for ASR, TTS, and Audio use cases
- `docs/PROMPTS.md` — Prompt templates for NeMo model workflows and integrations
- `agents/` — AI agent skills for NeMo speech AI workflows:
  - `agents/asr-agent/` — ASR transcription agent (single file, batch, multilingual)
  - `agents/tts-agent/` — TTS synthesis agent (single and batch generation)
  - `agents/audio-agent/` — Audio analysis agent (meeting analysis, quality check)
- `SECURITY.md` — Security policy and responsible disclosure guidelines
- `CODE_OF_CONDUCT.md` — Contributor Covenant Code of Conduct v2.1
- `CHANGELOG.md` — This changelog file

---

## [2.6.0] — 2025-11

### Added
- Support for Nemotron-3-Nano-30B-A3B-BF16 model
- NGC container version 25.11.nemotron_3_nano

### Changed
- Deprecated NeMo 2.0 LLM/VLM support in favor of [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [NeMo AutoModel](https://github.com/NVIDIA-NeMo/AutoModel)

---

## [2.5.0] — 2025-09

### Added
- Blackwell GPU support (GB200, B200)
- HuggingFace AutoModel integration for CausalLM and ImageTextToText models

---

## [2.4.0] — 2025-05

### Added
- Support for Llama 4, Flux, Llama Nemotron, Hyena/Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3

---

## [2.0.0] — 2024-11

### Added
- NeMo 2.0 architecture with Python-based configuration (replacing YAML)
- PyTorch Lightning modular abstractions
- NeMo-Run for multi-GPU/multi-node experiment management
- Megatron Core integration

---

*For the full NVIDIA NeMo upstream changelog, see [NVIDIA/NeMo releases](https://github.com/NVIDIA/NeMo/releases).*
