# Record for Agent — Local Audio Skill (`Zero-API ASR + Speaker Diarization`)

[中文](#中文说明) | [English](#english)

---

## 中文说明

面向 **Antigravity / Codex / Claude Code / OpenClaw** 等本地 AI Agent 的开源配套 Skill：无需任何付费语音 API、无需 HuggingFace Token、无需显卡，一键离线消费 **Record for Agent** 同步的录音、重点标星（`markers`）、录制地点（`location`）与 14 天 Apple 健康心率/压力基线（`biometrics`）。

### 内置零 API 开源模型栈（首次运行自动从官方 Release 拉取，共 ~210 MB）
- **原语言转写**：`sherpa-onnx` + `SenseVoice-Small (int8)` + `Silero-VAD`（原生支持中/英/日/粤/韩及中英混说，不擅自翻译）
- **离线声纹分离**：`sherpa-onnx` + `pyannote-segmentation-3-0` ONNX + `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`（按整场连续录音聚类，自动标出一致的 `说话人 A / B / C`，**零 HF Token**）

### 快速上手

```bash
git clone https://github.com/BENZEMA216/record-for-agent-skill.git
cd record-for-agent-skill
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. 自检云盘/本地收件箱连接状态与全量覆盖范围
python3 scripts/rfa_context.py check --dir ~/.local/share/context-hub-v2/inbox/audio

# 2. 一次性下载离线转写 + 声纹模型（~210 MB，来自 k2-fsa/sherpa-onnx 官方 Release）
python3 scripts/rfa_context.py setup

# 3. 离线转写 + 声纹分离（只读原文件，结果缓存于 ~/.cache/record-for-agent/）
python3 scripts/rfa_context.py transcribe --dir ~/.local/share/context-hub-v2/inbox/audio

# 4. 导出含重点标星、心率基线与说话人标签的 Markdown
python3 scripts/rfa_context.py export --dir ~/.local/share/context-hub-v2/inbox/audio --limit 5
```

> 仓库内**不包含**任何模型二进制权重，所有模型均在 `setup` 时直接从 [`k2-fsa/sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx/releases) 官方 Release 下载到 `~/.cache/record-for-agent/models/`。

---

## English

A lightweight, zero-API, zero-HF-token Agent Skill (`SKILL.md` + Python CLI) for **Antigravity, Codex, Claude Code, and OpenClaw** to ingest, transcribe, speaker-diarize, and analyze personal voice recordings from **Record for Agent**.

### Local Open-Source Stack (~210 MB auto-downloaded once from official `k2-fsa/sherpa-onnx` Releases)
- **Original-Language ASR**: `SenseVoice-Small (int8)` + `Silero-VAD` (Chinese / English / Japanese / Cantonese / Korean, code-switching preserved, zero auto-translation)
- **Offline Speaker Diarization**: `pyannote-segmentation-3-0` ONNX + `3D-Speaker ERes2Net` ONNX (session-level voiceprint clustering into `Speaker A / B / C` across 1-minute segments, **no HuggingFace token required**)
- **Read-Only Guarantee**: Source sync folders are never modified; transcripts cache cleanly under `~/.cache/record-for-agent/`.

### Usage

```bash
git clone https://github.com/BENZEMA216/record-for-agent-skill.git
cd record-for-agent-skill
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

python3 scripts/rfa_context.py check --dir ~/.local/share/context-hub-v2/inbox/audio
python3 scripts/rfa_context.py setup
python3 scripts/rfa_context.py transcribe --dir ~/.local/share/context-hub-v2/inbox/audio
python3 scripts/rfa_context.py export --dir ~/.local/share/context-hub-v2/inbox/audio --limit 5
```
