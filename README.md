# Record for Agent — Local Audio Skill (`Zero-API ASR + Dual-Factor Speaker Diarization`)

[![Download on the App Store](https://img.shields.io/badge/App%20Store-Record%20for%20Agent%20(Free)-0D96F6?logo=apple&logoColor=white)](https://apps.apple.com/app/id6831018336)
[![CI](https://github.com/rawmemory/record-for-agent-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/rawmemory/record-for-agent-skill/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security & Trust Envelope](https://img.shields.io/badge/Security-SHA--256%20Verified%20%26%20Offline-blue.svg)](SECURITY.md)

[📱 App Store 下载 (Free)](https://apps.apple.com/app/id6831018336) · [中文](#中文说明) · [English](#english) · [安全边界 / Security](SECURITY.md) · [💬 反馈与 Bug 收集](#反馈与-bug-收集--feedback--bug-reports)

> **Turn your Apple Watch into a private AI voice memory pipe — no \$159 hardware pendant, no monthly subscription, zero third-party relay servers.**
> **把你的 Apple Watch 变成私人 AI 语音记忆入口——无需千元录音硬件挂件、免费无订阅、零第三方中转服务器。**
> 👉 **[Get Record for Agent free on the App Store (174 countries & regions)](https://apps.apple.com/app/id6831018336)**

---

## 中文说明

面向 **Antigravity / Codex / Claude Code / OpenClaw** 等本地 AI Agent 的开源配套 Skill：无需任何付费语音 API、无需 HuggingFace Token、无需显卡，一键离线消费 **[Record for Agent](https://apps.apple.com/app/id6831018336)** 同步的录音、重点标星（`markers`）、录制地点（`location`）与 14 天 Apple 健康心率/压力基线（`biometrics`）。

### 双因子说话人识别（声纹 + 大模型语境联合决策）
- **因子 1 · 零 API 声纹聚类**：`sherpa-onnx` + `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` 对 `>=1.2s` 纯净人声提取 512 维向量，采用平均连接余弦聚类（自动剔除 `<|BGM|>` 背景广告与短插话过切），并输出簇间相似度与跨天本地声纹匹配分；
- **因子 2 · LLM 语境判断 + 毫秒级重聚类**：语言模型先通读对话语境（称呼、问答轮次、独白 vs 多人讨论）判断真实人数 `K`，若与纯声纹结果有出入，可 `<0.1s` 执行 `recluster --num-speakers K` 直接在缓存向量上重聚，并用语境校验边界句；
- **跨天本地声纹记忆**：用户确认一次“这是我 / 某某”后执行 `enroll`，后续不同日期的录音自动识别同一说话人，全部数据只存本机 `~/.cache/record-for-agent/`。

### 零 API 开源模型栈（首次 `setup` 自动从官方 Release 拉取，共 ~203 MB）
- **原语言转写**：`sherpa-onnx` + `SenseVoice-Small (int8)` + `Silero-VAD`（原生支持中/英/日/粤/韩及中英混说，不擅自翻译，透出 `<|BGM|>` 等事件标签）
- **声纹特征提取**：`3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`（~39.6 MB，**零 HF Token**）

### 快速上手

```bash
git clone https://github.com/rawmemory/record-for-agent-skill.git
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

A lightweight, zero-API, zero-HF-token Agent Skill (`SKILL.md` + Python CLI) for **Antigravity, Codex, Claude Code, and OpenClaw** to ingest, transcribe, run **Dual-Factor (Voiceprint + LLM Conversational Context) speaker diarization**, and analyze recordings captured with **[Record for Agent](https://apps.apple.com/app/id6831018336)** (free on the App Store for Apple Watch & iPhone).

### Dual-Factor Speaker Identification
- **Acoustic factor**: `3D-Speaker ERes2Net` ONNX (`sherpa-onnx`) extracts 512-d embeddings on clean speech turns (`>=1.2s`, excluding `<|BGM|>`), clusters via average-linkage cosine similarity, and matches against optional local cross-day voiceprints (`~/.cache/record-for-agent/voiceprints.json`).
- **Semantic factor**: The Agent reads conversational context (Q&A turns, direct address, monologue vs debate) to estimate speaker count `K`, runs instant sub-second `recluster --num-speakers K` on cached embeddings whenever needed, and fuses voiceprint scores with dialogue cues.
- **Read-only on source files**: Original sync directories are never modified.

### Usage

```bash
git clone https://github.com/rawmemory/record-for-agent-skill.git
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

---

## 反馈与 Bug 收集 / Feedback & Bug Reports

我们非常欢迎任何关于 **Apple Watch / iPhone App** 或 **本地 Skill** 的体验反馈、吐槽、Bug 报告与功能建议（中英文皆可，哪怕一句话也非常有帮助）：

- 🐛 **反馈 Bug / Report a Bug**：[新建 Bug Issue](https://github.com/rawmemory/record-for-agent-skill/issues/new?template=bug_report.yml)
- 💡 **提功能建议 / Feature Idea**：[新建功能建议](https://github.com/rawmemory/record-for-agent-skill/issues/new?template=feature_request.yml)
- 💬 **随手聊聊体验 / Discussions & Quick Feedback**：[GitHub Discussions](https://github.com/rawmemory/record-for-agent-skill/discussions) · [一句话快速反馈](https://github.com/rawmemory/record-for-agent-skill/issues/new?template=feedback.yml)
- 📧 **邮件直联 / Direct Support**：[recordforagent@gmail.com](mailto:recordforagent@gmail.com) · [帮助与隐私政策](https://record-for-agent.benzemazdz99.chatgpt.site/support/)

> 🔒 **隐私提示**：反馈问题时请勿上传真实个人录音或云盘凭据；如需附带排查信息，可在 iPhone App「设置 → 日志与诊断」一键导出脱敏诊断信息。
