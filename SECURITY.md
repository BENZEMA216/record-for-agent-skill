# Security & Trust Envelope / 安全与权限边界声明

This skill handles private personal voice recordings. Every design decision prioritizes **verifiable transparency, local-first execution, and zero unexpected network or filesystem access**.

本 Skill 处理高度私密的个人语音数据，坚持**明文可审计、本地离线执行、最小权限边界**原则。

---

## 1. Network Access / 网络边界

| Command | Network Access | Destination |
|---|---|---|
| `setup` | Read-only HTTPS download (one-time) | `https://github.com/k2-fsa/sherpa-onnx/releases` (3 pinned artifacts, strict SHA-256 verified) |
| `check`, `list`, `transcribe`, `recluster`, `enroll`, `voiceprints`, `export` | **None (100% offline)** | No telemetry, no analytics, no cloud ASR upload, no external APIs |

### Pinned Model SHA-256 Checksums

`scripts/rfa_context.py` verifies the exact SHA-256 digest of every downloaded model before extracting or saving it; any mismatch fails closed and deletes the temporary file:

| Artifact | Size | SHA-256 |
|---|---|---|
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2` | 163,002,883 B | `7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e` |
| `silero_vad.onnx` | 643,854 B | `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6` |
| `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | 39,593,762 B | `f1ee61be04000b50392a63442e5dd9feaea7e81dddd494d3647cf1c3d88a1498` |

---

## 2. Filesystem & Credential Boundaries / 文件与密钥边界

- **Source sync directories are strictly read-only**: `rfa_context.py` never writes, renames, or deletes any file inside `~/.local/share/context-hub-v2/inbox/audio/` or user-provided `--dir` folders.
- **Only local write target**: `~/.cache/record-for-agent/` (downloaded ONNX models, cached transcripts/embeddings, and optional local speaker voiceprint centroids in `voiceprints.json`), overridable via `RFA_MODELS_DIR`, `RFA_TRANSCRIPTS_DIR`, and `RFA_VOICEPRINTS_PATH`.
- **Zero credential harvesting**: Never reads Keychain, SSH keys, browser cookies, or unrelated environment variables; never executes remote shell scripts (`curl | bash`), `eval`, `exec`, or `subprocess(..., shell=True)`.
- **Prompt-injection defense**: `SKILL.md` instructs Agents to treat all audio, transcripts, and JSON metadata strictly as untrusted reference data—never as executable instructions.

---

## 3. Verifying Before Running / 运行前快速审计

The entire codebase is intentionally tiny (1 Markdown skill + 1 self-contained Python file) and covered by automated CI checks:

```bash
# Pin to an immutable release tag instead of tracking `main`
git clone --branch v0.2.0 --depth 1 https://github.com/BENZEMA216/record-for-agent-skill.git
```

To report a security issue, please open a private security advisory on GitHub (`Security -> Advisories -> Report a vulnerability`).
