# Record for Agent — Local Audio Skill (`Zero-API ASR + Dual-Factor Speaker Diarization`)

[![CI](https://github.com/BENZEMA216/record-for-agent-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/BENZEMA216/record-for-agent-skill/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security & Trust Envelope](https://img.shields.io/badge/Security-SHA--256%20Verified%20%26%20Offline-blue.svg)](SECURITY.md)

[中文](#中文说明) | [English](#english) | [安全边界 / Security](SECURITY.md)

---

## 中文说明

面向 **Antigravity / Codex / Claude Code / OpenClaw** 等本地 AI Agent 的开源配套 Skill：无需任何付费语音 API、无需 HuggingFace Token、无需显卡，一键离线消费 **Record for Agent** 同步的录音、重点标星（`markers`）、录制地点（`location`）与 14 天 Apple 健康心率/压力基线（`biometrics`）。

### 双因子说话人识别（声纹 + 大模型语境联合决策）
- **因子 1 · 零 API 声纹聚类**：`sherpa-onnx` + `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` 对 `>=1.2s` 纯净人声提取 512 维向量，采用平均连接余弦聚类（自动剔除 `<|BGM|>` 背景广告与短插话过切），并输出簇间相似度与跨天本地声纹匹配分；
- **因子 2 · LLM 语境判断 + 毫秒级重聚类**：语言模型先通读对话语境（称呼、问答轮次、独白 vs 多人讨论）判断真实人数 `K`，若与纯声纹结果有出入，可 `<0.1s` 执行 `recluster --num-speakers K` 直接在缓存向量上重聚，并用语境校验边界句；
- **跨天本地声纹记忆**：用户确认一次“这是我 / 某某”后执行 `enroll`，后续不同日期的录音自动识别同一说话人，全部数据只存本机 `~/.cache/record-for-agent/`。

### 零 API 开源模型栈（首次 `setup` 自动从官方 Release 拉取，共 ~203 MB）
- **原语言转写**：`sherpa-onnx` + `SenseVoice-Small (int8)` + `Silero-VAD`（原生支持中/英/日/粤/韩及中英混说，不擅自翻译，透出 `<|BGM|>` 等事件标签）
- **声纹特征提取**：`3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`（~39.6 MB，**零 HF Token**）

### 快速上手

```bash
git clone https://github.com/BENZEMA216/record-for-agent-skill.git
cd record-for-agent-skill
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. 自检云盘/本地收件箱连接状态与全量覆盖范围
python3 scripts/rfa_context.py check --dir ~/.local/share/context-hub-v2/inbox/audio

# 2. 一次性下载离线转写 + 声纹模型（~203 MB，来自 k2-fsa/sherpa-onnx 官方 Release）
python3 scripts/rfa_context.py setup

# 3. 离线原语言转写 + 3D-Speaker 声纹聚类（原文件只读，结果缓存于 ~/.cache/record-for-agent/）
python3 scripts/rfa_context.py transcribe --dir ~/.local/share/context-hub-v2/inbox/audio --limit 3

# 4. LLM 结合语境判断真实人数 K 后，毫秒级重聚类（无需重跑转写）
python3 scripts/rfa_context.py recluster --group <GROUP_ID_PREFIX> --num-speakers 2

# 5. 保存已确认说话人声纹，供后续跨天自动认人
python3 scripts/rfa_context.py enroll --group <GROUP_ID_PREFIX> --speaker "Speaker A" --name "我"

# 6. 导出含重点标星、心率基线与说话人标签的 Markdown
python3 scripts/rfa_context.py export --dir ~/.local/share/context-hub-v2/inbox/audio --limit 3
```

---

## English

A lightweight, zero-API, zero-HF-token Agent Skill (`SKILL.md` + Python CLI) for **Antigravity, Codex, Claude Code, and OpenClaw** to ingest, transcribe, run **Dual-Factor (Voiceprint + LLM Conversational Context) speaker diarization**, and analyze recordings from **Record for Agent**.

### Dual-Factor Speaker Identification
- **Acoustic factor**: `3D-Speaker ERes2Net` ONNX (`sherpa-onnx`) extracts 512-d embeddings on clean speech turns (`>=1.2s`, excluding `<|BGM|>`), clusters via average-linkage cosine similarity, and matches against optional local cross-day voiceprints (`~/.cache/record-for-agent/voiceprints.json`).
- **Semantic factor**: The Agent reads conversational context (Q&A turns, direct address, monologue vs debate) to estimate speaker count `K`, runs instant sub-second `recluster --num-speakers K` on cached embeddings whenever needed, and fuses voiceprint scores with dialogue cues.
- **Read-only on source files**: Original sync directories are never modified.

### Usage

```bash
git clone https://github.com/BENZEMA216/record-for-agent-skill.git
cd record-for-agent-skill
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

python3 scripts/rfa_context.py check --dir ~/.local/share/context-hub-v2/inbox/audio
python3 scripts/rfa_context.py setup
python3 scripts/rfa_context.py transcribe --dir ~/.local/share/context-hub-v2/inbox/audio --limit 3
python3 scripts/rfa_context.py recluster --group <GROUP_ID_PREFIX> --num-speakers 2
python3 scripts/rfa_context.py enroll --group <GROUP_ID_PREFIX> --speaker "Speaker A" --name "Me"
python3 scripts/rfa_context.py export --dir ~/.local/share/context-hub-v2/inbox/audio --limit 3
```
