#!/usr/bin/env python3
"""Zero-API local ingestion, original-language ASR, and speaker diarization for Record for Agent.

Supports:
- iCloud Mac Companion inbox (`~/.local/share/context-hub-v2/inbox/audio/`: `<id>.m4a` + `<id>.json`)
- Google Drive synced folders (`<prefix>.manifest.json` + `<prefix>.m4a`/`.caf` + `.transcript-*`)
- WebDAV atomic packets (`*.ctxaudio.zip` containing `manifest.json` + `.m4a`)

Never modifies, renames, or deletes source files. Cached transcripts and ONNX models
live strictly under `~/.cache/record-for-agent/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import urllib.request
import wave
import zipfile
from zoneinfo import ZoneInfo

DEFAULT_INBOX_DIR = Path.home() / ".local" / "share" / "context-hub-v2" / "inbox" / "audio"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "record-for-agent"
CONTEXT_HUB_MODELS_DIR = Path.home() / ".local" / "share" / "context-hub-v2" / "models"

SENSEVOICE_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SENSEVOICE_FALLBACK_DIRS = (
    SENSEVOICE_DIR_NAME,
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
)
SENSEVOICE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{SENSEVOICE_DIR_NAME}.tar.bz2"
)
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)
PYANNOTE_SEG_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
PYANNOTE_SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
    f"{PYANNOTE_SEG_DIR}.tar.bz2"
)
SPEAKER_EMBED_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
SPEAKER_EMBED_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    f"{SPEAKER_EMBED_NAME}"
)


# ---------------------------------------------------------------------------
# Data Models & Parsing
# ---------------------------------------------------------------------------


@dataclass
class SegmentRecord:
    segment_id: str
    group_id: str
    sequence: int
    started_at_ms: int
    ended_at_ms: int
    timezone_id: str | None
    sha256: str
    byte_count: int
    audio_path: Path | None = None
    zip_packet_path: Path | None = None
    zip_audio_member: str | None = None
    manifest_path: Path | None = None
    existing_transcript: str | None = None
    location: dict[str, Any] | None = None
    biometrics: dict[str, Any] | None = None
    markers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.ended_at_ms - self.started_at_ms) / 1000.0)


@dataclass
class RecordingGroup:
    group_id: str
    segments: list[SegmentRecord]

    @property
    def started_at_ms(self) -> int:
        return min(s.started_at_ms for s in self.segments)

    @property
    def ended_at_ms(self) -> int:
        return max(s.ended_at_ms for s in self.segments)

    @property
    def timezone_id(self) -> str | None:
        for s in self.segments:
            if s.timezone_id:
                return s.timezone_id
        return None

    @property
    def duration_seconds(self) -> float:
        return sum(s.duration_seconds for s in self.segments)

    @property
    def marker_count(self) -> int:
        return sum(len(s.markers) for s in self.segments)

    @property
    def has_biometrics(self) -> bool:
        return any(s.biometrics is not None for s in self.segments)

    @property
    def has_location(self) -> bool:
        return any(s.location is not None for s in self.segments)

    @property
    def has_existing_transcript(self) -> bool:
        return any(bool((s.existing_transcript or "").strip()) for s in self.segments)

    def local_date_str(self) -> str:
        return format_timestamp(self.started_at_ms, self.timezone_id)[:10]


def parse_timestamp_ms(value: Any) -> int | None:
    """Parse epoch milliseconds, epoch seconds, Apple CFAbsoluteTime, or ISO-8601 strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if num > 1e11:  # already milliseconds
            return int(round(num))
        if num > 1e9:  # Unix seconds
            return int(round(num * 1000))
        # Apple CFAbsoluteTime (seconds since 2001-01-01)
        return int(round((num + 978307200.0) * 1000))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(round(dt.timestamp() * 1000))
        except ValueError:
            return None
    return None


def format_timestamp(ms: int, tz_id: str | None = None) -> str:
    dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    if tz_id:
        try:
            return dt_utc.astimezone(ZoneInfo(tz_id)).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            pass
    return dt_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _parse_manifest_dict(
    data: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    audio_path: Path | None = None,
    zip_packet_path: Path | None = None,
    zip_audio_member: str | None = None,
    external_transcript: str | None = None,
) -> SegmentRecord | None:
    seg_id = str(data.get("id") or "").strip().lower()
    if not seg_id:
        return None

    started_ms = parse_timestamp_ms(data.get("startedAt"))
    ended_ms = parse_timestamp_ms(data.get("endedAt"))
    if started_ms is None:
        return None
    if ended_ms is None or ended_ms < started_ms:
        ended_ms = started_ms

    rec_group = data.get("recordingGroup")
    group_id = ""
    if isinstance(rec_group, dict) and rec_group.get("id"):
        group_id = str(rec_group["id"]).strip().lower()
    elif data.get("recordingGroupID"):
        group_id = str(data["recordingGroupID"]).strip().lower()
    elif data.get("sessionID"):
        group_id = str(data["sessionID"]).strip().lower()
    else:
        group_id = seg_id

    try:
        sequence = int(data.get("sequence", 0))
    except (TypeError, ValueError):
        sequence = 0

    sha = str(data.get("sha256") or "").strip().lower()
    try:
        byte_count = int(data.get("byteCount", 0))
    except (TypeError, ValueError):
        byte_count = 0

    transcript = data.get("transcript")
    if isinstance(transcript, dict):
        transcript = transcript.get("text")
    if not isinstance(transcript, str) or not transcript.strip():
        transcript = external_transcript

    markers = data.get("markers")
    if not isinstance(markers, list):
        markers = []

    biometrics = data.get("biometrics") if isinstance(data.get("biometrics"), dict) else None
    location = data.get("location") if isinstance(data.get("location"), dict) else None

    return SegmentRecord(
        segment_id=seg_id,
        group_id=group_id,
        sequence=sequence,
        started_at_ms=started_ms,
        ended_at_ms=ended_ms,
        timezone_id=data.get("timeZoneIdentifier") if isinstance(data.get("timeZoneIdentifier"), str) else None,
        sha256=sha,
        byte_count=byte_count,
        audio_path=audio_path,
        zip_packet_path=zip_packet_path,
        zip_audio_member=zip_audio_member,
        manifest_path=manifest_path,
        existing_transcript=transcript.strip() if isinstance(transcript, str) and transcript.strip() else None,
        location=location,
        biometrics=biometrics,
        markers=[m for m in markers if isinstance(m, dict)],
    )


def _find_matching_external_transcript(parent: Path, prefix: str) -> str | None:
    for pattern in (f"{prefix}.transcript-*.txt", f"{prefix}.transcript-*.json", f"{prefix}.txt"):
        for candidate in sorted(parent.glob(pattern)):
            try:
                raw = candidate.read_text(encoding="utf-8").strip()
                if not raw:
                    continue
                if candidate.suffix.lower() == ".json":
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                        return parsed["text"].strip()
                else:
                    return raw
            except Exception:
                continue
    return None


def _find_matching_audio(parent: Path, prefix: str, file_name: str | None) -> Path | None:
    if file_name:
        direct = parent / file_name
        if direct.is_file():
            return direct
    for ext in (".m4a", ".caf", ".wav", ".aac", ".mp3"):
        candidate = parent / f"{prefix}{ext}"
        if candidate.is_file():
            return candidate
    return None


def discover_groups(directories: list[Path]) -> tuple[list[RecordingGroup], dict[str, Any]]:
    """Scan one or more sync directories read-only, deduplicate by (id, sha256), and group sessions."""
    by_key: dict[tuple[str, str], SegmentRecord] = {}
    stats: dict[str, Any] = {
        "scanned_dirs": [],
        "missing_dirs": [],
        "json_manifests_seen": 0,
        "zip_packets_seen": 0,
        "duplicates_merged": 0,
    }

    for root in directories:
        expanded = root.expanduser()
        if not expanded.exists() or not expanded.is_dir():
            stats["missing_dirs"].append(str(expanded))
            continue
        stats["scanned_dirs"].append(str(expanded))

        for path in sorted(expanded.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue

            lower = path.name.lower()
            seg: SegmentRecord | None = None

            if lower.endswith(".ctxaudio.zip"):
                stats["zip_packets_seen"] += 1
                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        names = zf.namelist()
                        manifest_name = next((n for n in names if n.endswith("manifest.json")), None)
                        audio_name = next(
                            (n for n in names if n.lower().endswith((".m4a", ".caf", ".wav"))),
                            None,
                        )
                        if manifest_name:
                            data = json.loads(zf.read(manifest_name).decode("utf-8"))
                            if isinstance(data, dict):
                                seg = _parse_manifest_dict(
                                    data,
                                    manifest_path=path,
                                    zip_packet_path=path,
                                    zip_audio_member=audio_name,
                                )
                except Exception:
                    continue

            elif lower.endswith(".json") and not lower.endswith(".rfa-transcript.json") and ".transcript-" not in lower:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict) or "startedAt" not in data or "id" not in data:
                    continue
                stats["json_manifests_seen"] += 1
                if lower.endswith(".manifest.json"):
                    prefix = path.name[: -len(".manifest.json")]
                else:
                    prefix = path.stem
                audio_path = _find_matching_audio(
                    path.parent,
                    prefix,
                    data.get("fileName") if isinstance(data.get("fileName"), str) else None,
                )
                ext_tx = _find_matching_external_transcript(path.parent, prefix)
                seg = _parse_manifest_dict(
                    data,
                    manifest_path=path,
                    audio_path=audio_path,
                    external_transcript=ext_tx,
                )

            if seg is None:
                continue

            dedup_key = (seg.segment_id, seg.sha256 or seg.segment_id)
            existing = by_key.get(dedup_key)
            if existing is None:
                by_key[dedup_key] = seg
            else:
                stats["duplicates_merged"] += 1
                # Prefer the copy that has local audio / transcript / biometrics / markers
                if existing.audio_path is None and seg.audio_path is not None:
                    existing.audio_path = seg.audio_path
                if existing.zip_packet_path is None and seg.zip_packet_path is not None:
                    existing.zip_packet_path = seg.zip_packet_path
                    existing.zip_audio_member = seg.zip_audio_member
                if not existing.existing_transcript and seg.existing_transcript:
                    existing.existing_transcript = seg.existing_transcript
                if existing.biometrics is None and seg.biometrics is not None:
                    existing.biometrics = seg.biometrics
                if not existing.markers and seg.markers:
                    existing.markers = seg.markers

    grouped: dict[str, list[SegmentRecord]] = {}
    for seg in by_key.values():
        grouped.setdefault(seg.group_id, []).append(seg)

    groups: list[RecordingGroup] = []
    for gid, segs in grouped.items():
        segs.sort(key=lambda s: (s.sequence, s.started_at_ms, s.segment_id))
        groups.append(RecordingGroup(group_id=gid, segments=segs))
    groups.sort(key=lambda g: g.started_at_ms)
    return groups, stats


# ---------------------------------------------------------------------------
# Model Resolution & Zero-API Download
# ---------------------------------------------------------------------------


def default_models_dir() -> Path:
    if env := os.environ.get("RFA_MODELS_DIR"):
        return Path(env).expanduser()
    return DEFAULT_CACHE_ROOT / "models"


def default_transcript_cache_dir() -> Path:
    if env := os.environ.get("RFA_TRANSCRIPTS_DIR"):
        return Path(env).expanduser()
    return DEFAULT_CACHE_ROOT / "transcripts"


def resolve_model_paths(models_dir: Path | None = None) -> dict[str, Any]:
    primary = (models_dir or default_models_dir()).expanduser()
    search_roots = [primary]
    if CONTEXT_HUB_MODELS_DIR != primary and CONTEXT_HUB_MODELS_DIR.is_dir():
        search_roots.append(CONTEXT_HUB_MODELS_DIR)

    sv_model: Path | None = None
    sv_tokens: Path | None = None
    vad_path: Path | None = None
    seg_model: Path | None = None
    embed_model: Path | None = None

    for root in search_roots:
        if sv_model is None:
            for dname in SENSEVOICE_FALLBACK_DIRS:
                cand = root / dname
                tokens = cand / "tokens.txt"
                model = cand / "model.int8.onnx"
                if not model.is_file():
                    model = cand / "model.onnx"
                if model.is_file() and tokens.is_file():
                    sv_model, sv_tokens = model, tokens
                    break
        if vad_path is None and (root / "silero_vad.onnx").is_file():
            vad_path = root / "silero_vad.onnx"
        if seg_model is None:
            cand = root / PYANNOTE_SEG_DIR / "model.onnx"
            if cand.is_file():
                seg_model = cand
        if embed_model is None:
            cand = root / SPEAKER_EMBED_NAME
            if cand.is_file():
                embed_model = cand

    return {
        "models_dir": str(primary),
        "sensevoice_ready": sv_model is not None and sv_tokens is not None,
        "sensevoice_model": str(sv_model) if sv_model else None,
        "sensevoice_tokens": str(sv_tokens) if sv_tokens else None,
        "silero_vad_ready": vad_path is not None,
        "silero_vad_path": str(vad_path) if vad_path else None,
        "diarization_ready": seg_model is not None and embed_model is not None,
        "segmentation_model": str(seg_model) if seg_model else None,
        "embedding_model": str(embed_model) if embed_model else None,
    }


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "record-for-agent-skill/0.1"})
    print(f"Downloading {url} -> {destination.name} ...", file=sys.stderr, flush=True)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=256 * 1024)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_extract_tar_bz2(archive_path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    with tarfile.open(archive_path, "r:bz2") as tar:
        members = []
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(resolved_root):
                raise ValueError(f"Unsafe tar member path: {member.name}")
            members.append(member)
        if sys.version_info >= (3, 12):
            tar.extractall(root, members=members, filter="data")
        else:
            tar.extractall(root, members=members)


def download_models(models_dir: Path | None = None, *, include_diarization: bool = True) -> dict[str, Any]:
    root = (models_dir or default_models_dir()).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    status = resolve_model_paths(root)

    if not status["sensevoice_ready"]:
        archive = root / f"{SENSEVOICE_DIR_NAME}.tar.bz2"
        _download_file(SENSEVOICE_URL, archive)
        try:
            _safe_extract_tar_bz2(archive, root)
        finally:
            archive.unlink(missing_ok=True)

    if not status["silero_vad_ready"]:
        _download_file(SILERO_VAD_URL, root / "silero_vad.onnx")

    if include_diarization and not status["diarization_ready"]:
        if not status["segmentation_model"]:
            seg_archive = root / f"{PYANNOTE_SEG_DIR}.tar.bz2"
            _download_file(PYANNOTE_SEG_URL, seg_archive)
            try:
                _safe_extract_tar_bz2(seg_archive, root)
            finally:
                seg_archive.unlink(missing_ok=True)
        if not status["embedding_model"]:
            _download_file(SPEAKER_EMBED_URL, root / SPEAKER_EMBED_NAME)

    return resolve_model_paths(root)


# ---------------------------------------------------------------------------
# Audio Decode, Original-Language ASR, and Speaker Diarization
# ---------------------------------------------------------------------------


def decode_audio_to_pcm16k(audio_path: Path) -> tuple[Any, int]:
    """Decode `.m4a`/`.caf`/`.wav`/`.aac` into 16 kHz mono float32 numpy array."""
    import numpy as np

    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000:
                    raw = wf.readframes(wf.getnframes())
                    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, 16000
        except wave.Error:
            pass

    with tempfile.NamedTemporaryFile(suffix=".wav", prefix="rfa-pcm-", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        afconvert = shutil.which("afconvert")
        ffmpeg = shutil.which("ffmpeg")
        if afconvert:
            cmd = [afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(audio_path), str(wav_path)]
        elif ffmpeg:
            cmd = [ffmpeg, "-y", "-v", "error", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path)]
        else:
            raise RuntimeError("Neither macOS `afconvert` nor `ffmpeg` is available to decode audio.")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"Audio decode failed ({proc.returncode}): {proc.stderr.strip()[:240]}")

        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr
    finally:
        wav_path.unlink(missing_ok=True)


def decode_segment_audio(seg: SegmentRecord) -> tuple[Any, int] | None:
    if seg.audio_path and seg.audio_path.is_file():
        return decode_audio_to_pcm16k(seg.audio_path)
    if seg.zip_packet_path and seg.zip_audio_member and seg.zip_packet_path.is_file():
        suffix = Path(seg.zip_audio_member).suffix or ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix="rfa-zip-", delete=False) as tmp:
            extracted = Path(tmp.name)
        try:
            with zipfile.ZipFile(seg.zip_packet_path, "r") as zf:
                extracted.write_bytes(zf.read(seg.zip_audio_member))
            return decode_audio_to_pcm16k(extracted)
        finally:
            extracted.unlink(missing_ok=True)
    return None


def speaker_index_to_label(index: int) -> str:
    """Map integer speaker cluster 0,1,2... -> 'Speaker A', 'Speaker B', ..."""
    if 0 <= index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def assign_speaker_to_span(
    span_start: float,
    span_end: float,
    diar_segments: list[tuple[float, float, int]],
) -> str | None:
    """Assign speaker label with maximum time overlap against a diarization timeline."""
    if not diar_segments:
        return None
    overlap_by_speaker: dict[int, float] = {}
    for d_start, d_end, spk in diar_segments:
        ov = min(span_end, d_end) - max(span_start, d_start)
        if ov > 0:
            overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + ov
    if not overlap_by_speaker:
        # Fallback: nearest speaker segment within 1.5s
        mid = 0.5 * (span_start + span_end)
        best_spk = None
        best_dist = 1.5
        for d_start, d_end, spk in diar_segments:
            dist = 0.0 if d_start <= mid <= d_end else min(abs(mid - d_start), abs(mid - d_end))
            if dist < best_dist:
                best_dist = dist
                best_spk = spk
        return speaker_index_to_label(best_spk) if best_spk is not None else None
    best_spk = max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]
    return speaker_index_to_label(best_spk)


class LocalAudioPipeline:
    """Runs SenseVoice-Small (auto spoken language) + Silero-VAD + optional 3D-Speaker diarization."""

    def __init__(
        self,
        models_dir: Path | None = None,
        *,
        enable_diarization: bool = True,
        num_threads: int = 4,
        diarization_threshold: float = 0.5,
    ):
        self.models_dir = models_dir
        self.enable_diarization = enable_diarization
        self.num_threads = max(1, num_threads)
        self.diarization_threshold = diarization_threshold
        self._recognizer: Any = None
        self._vad_config: Any = None
        self._diarizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._recognizer is not None:
            return
        import sherpa_onnx

        status = resolve_model_paths(self.models_dir)
        if not status["sensevoice_ready"]:
            raise FileNotFoundError(
                f"Local ASR models not found under {status['models_dir']}. Run `python3 scripts/rfa_context.py setup` first."
            )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=status["sensevoice_model"],
            tokens=status["sensevoice_tokens"],
            num_threads=self.num_threads,
            sample_rate=16000,
            language="",  # Auto-detect real spoken language; never force translation
            use_itn=True,
            provider="cpu",
        )
        if status["silero_vad_ready"]:
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = status["silero_vad_path"]
            cfg.silero_vad.threshold = 0.45
            cfg.silero_vad.min_silence_duration = 0.45
            cfg.silero_vad.min_speech_duration = 0.25
            cfg.silero_vad.max_speech_duration = 22.0
            cfg.silero_vad.window_size = 512
            cfg.sample_rate = 16000
            cfg.num_threads = 1
            self._vad_config = cfg

        if self.enable_diarization and status["diarization_ready"]:
            sd_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=status["segmentation_model"]
                    ),
                    num_threads=self.num_threads,
                    provider="cpu",
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=status["embedding_model"],
                    num_threads=self.num_threads,
                    provider="cpu",
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=-1,
                    threshold=self.diarization_threshold,
                ),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(sd_cfg)

    def _vad_chunks(self, samples: Any, sr: int) -> list[tuple[float, float, Any]]:
        import numpy as np

        if self._vad_config is None:
            win = 20 * sr
            return [
                (i / sr, min(len(samples), i + win) / sr, samples[i : i + win])
                for i in range(0, len(samples), win)
                if len(samples[i : i + win]) >= sr // 4
            ]

        import sherpa_onnx

        vad = sherpa_onnx.VoiceActivityDetector(self._vad_config, buffer_size_in_seconds=120)
        window = self._vad_config.silero_vad.window_size
        out: list[tuple[float, float, Any]] = []
        for idx in range(0, len(samples), window):
            chunk = samples[idx : idx + window]
            if len(chunk) < window:
                padded = np.zeros(window, dtype=np.float32)
                padded[: len(chunk)] = chunk
                vad.accept_waveform(padded)
            else:
                vad.accept_waveform(chunk)
            while not vad.empty():
                seg = vad.front
                out.append((seg.start / sr, (seg.start + len(seg.samples)) / sr, seg.samples))
                vad.pop()
        vad.flush()
        while not vad.empty():
            seg = vad.front
            out.append((seg.start / sr, (seg.start + len(seg.samples)) / sr, seg.samples))
            vad.pop()
        return out

    def transcribe_group(self, group: RecordingGroup) -> dict[str, Any]:
        import numpy as np

        t0 = time.perf_counter()
        self._ensure_loaded()

        # Concatenate decoded PCM across the whole recording group so speaker labels stay consistent
        pcm_parts: list[Any] = []
        segment_spans: list[tuple[SegmentRecord, float, float]] = []
        cursor_s = 0.0
        for seg in group.segments:
            decoded = decode_segment_audio(seg)
            if decoded is None:
                continue
            samples, sr = decoded
            if sr != 16000 or len(samples) == 0:
                continue
            dur = len(samples) / 16000.0
            pcm_parts.append(samples)
            segment_spans.append((seg, cursor_s, cursor_s + dur))
            cursor_s += dur

        if not pcm_parts:
            return {
                "schema_version": 1,
                "group_id": group.group_id,
                "engine": "sensevoice-small",
                "diarized": False,
                "speaker_count": 0,
                "languages": [],
                "utterances": [],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            }

        full_pcm = np.concatenate(pcm_parts)
        peak = float(max(abs(full_pcm.min()), abs(full_pcm.max()))) if len(full_pcm) else 0.0
        if len(full_pcm) < 3200 or peak < 1e-4:
            return {
                "schema_version": 1,
                "group_id": group.group_id,
                "engine": "sensevoice-small",
                "diarized": False,
                "speaker_count": 0,
                "languages": [],
                "utterances": [],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            }

        # Step 1: Session-level offline speaker diarization (capped in 25-min windows for RAM safety)
        diar_segments: list[tuple[float, float, int]] = []
        diarized = False
        distinct_speakers: set[int] = set()
        if self._diarizer is not None and cursor_s >= 3.0:
            max_chunk_samples = 25 * 60 * 16000
            if len(full_pcm) <= max_chunk_samples:
                try:
                    raw_res = self._diarizer.process(full_pcm).sort_by_start_time()
                    for item in raw_res:
                        diar_segments.append((float(item.start), float(item.end), int(item.speaker)))
                        distinct_speakers.add(int(item.speaker))
                    diarized = True
                except Exception as exc:
                    print(f"Warning: diarization failed for {group.group_id}: {exc}", file=sys.stderr)

        # Remap arbitrary speaker IDs by order of first appearance -> 0, 1, 2...
        speaker_order: dict[int, int] = {}
        normalized_diar: list[tuple[float, float, int]] = []
        for s_start, s_end, raw_spk in diar_segments:
            if raw_spk not in speaker_order:
                speaker_order[raw_spk] = len(speaker_order)
            normalized_diar.append((s_start, s_end, speaker_order[raw_spk]))

        multi_speaker = len(speaker_order) >= 2

        # Step 2: Silero-VAD + SenseVoice-Small original-language ASR
        utterances: list[dict[str, Any]] = []
        languages: set[str] = set()
        for start_s, end_s, chunk in self._vad_chunks(full_pcm, 16000):
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, chunk)
            self._recognizer.decode_stream(stream)
            text = (stream.result.text or "").strip()
            if not text:
                continue
            lang = getattr(stream.result, "lang", "") or ""
            lang = lang.strip().strip("<|>")
            if lang:
                languages.add(lang)

            speaker = assign_speaker_to_span(start_s, end_s, normalized_diar) if multi_speaker else None
            # Resolve owning segment ID for precise traceability
            mid = 0.5 * (start_s + end_s)
            seg_id = group.segments[0].segment_id
            for seg, seg_start, seg_end in segment_spans:
                if seg_start <= mid <= seg_end:
                    seg_id = seg.segment_id
                    break

            utterances.append(
                {
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "speaker": speaker,
                    "lang": lang or None,
                    "segment_id": seg_id,
                    "text": text,
                }
            )

        return {
            "schema_version": 1,
            "group_id": group.group_id,
            "engine": "sensevoice-small",
            "diarized": diarized,
            "speaker_count": len(speaker_order) if diarized else (1 if utterances else 0),
            "languages": sorted(languages),
            "utterances": utterances,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }


# ---------------------------------------------------------------------------
# Cache & Rich Markdown Rendering
# ---------------------------------------------------------------------------


def load_cached_transcript(cache_dir: Path, group_id: str) -> dict[str, Any] | None:
    path = cache_dir / f"{group_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def save_cached_transcript(cache_dir: Path, group_id: str, payload: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{group_id}.json"
    tmp = cache_dir / f".tmp-{group_id}.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def _format_offset(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def summarize_group_biometrics(group: RecordingGroup) -> dict[str, Any] | None:
    avgs: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    baselines: list[float] = []
    scores: list[int] = []
    states: list[str] = []
    activities: list[str] = []
    for seg in group.segments:
        b = seg.biometrics
        if not b:
            continue
        if isinstance(b.get("heartRateAvg"), (int, float)):
            avgs.append(float(b["heartRateAvg"]))
        if isinstance(b.get("heartRateMin"), (int, float)):
            mins.append(float(b["heartRateMin"]))
        if isinstance(b.get("heartRateMax"), (int, float)):
            maxs.append(float(b["heartRateMax"]))
        if isinstance(b.get("restingBaselineBPM"), (int, float)):
            baselines.append(float(b["restingBaselineBPM"]))
        if isinstance(b.get("arousalScore"), (int, float)):
            scores.append(int(round(b["arousalScore"])))
        if isinstance(b.get("arousalState"), str):
            states.append(b["arousalState"])
        if isinstance(b.get("activityContext"), str):
            activities.append(b["activityContext"])
    if not avgs and not scores:
        return None
    avg_bpm = round(sum(avgs) / len(avgs), 1) if avgs else None
    base_bpm = round(sum(baselines) / len(baselines), 1) if baselines else None
    return {
        "heart_rate_avg": avg_bpm,
        "heart_rate_min": round(min(mins), 1) if mins else None,
        "heart_rate_max": round(max(maxs), 1) if maxs else None,
        "resting_baseline_bpm": base_bpm,
        "delta_bpm": round(avg_bpm - base_bpm, 1) if (avg_bpm is not None and base_bpm is not None) else None,
        "arousal_score_peak": max(scores) if scores else None,
        "arousal_states": sorted(set(states)),
        "activity_contexts": sorted(set(activities)),
    }


def render_group_markdown(
    group: RecordingGroup,
    cached: dict[str, Any] | None,
    speaker_map: dict[str, str],
) -> str:
    lines: list[str] = []
    start_str = format_timestamp(group.started_at_ms, group.timezone_id)
    lines.append(f"## {start_str} · {_format_offset(group.duration_seconds)} (`{group.group_id[:8]}`)")
    lines.append(f"- Segments: {len(group.segments)} (`{group.segments[0].segment_id}` … `{group.segments[-1].segment_id}`)")

    loc = next((s.location for s in group.segments if s.location), None)
    if loc:
        lat, lon = loc.get("latitude"), loc.get("longitude")
        acc = loc.get("horizontalAccuracy")
        if lat is not None and lon is not None:
            acc_text = f" (±{round(float(acc))}m)" if isinstance(acc, (int, float)) else ""
            lines.append(f"- Location: {float(lat):.5f}, {float(lon):.5f}{acc_text}")

    bio = summarize_group_biometrics(group)
    if bio:
        parts = []
        if bio["heart_rate_avg"] is not None:
            parts.append(f"HR avg {bio['heart_rate_avg']} bpm (min {bio['heart_rate_min']} / max {bio['heart_rate_max']})")
        if bio["resting_baseline_bpm"] is not None and bio["delta_bpm"] is not None:
            sign = "+" if bio["delta_bpm"] >= 0 else ""
            parts.append(f"14d baseline {bio['resting_baseline_bpm']} ({sign}{bio['delta_bpm']} bpm)")
        if bio["arousal_score_peak"] is not None:
            states = "/".join(bio["arousal_states"]) or "unknown"
            parts.append(f"arousal peak {bio['arousal_score_peak']}/100 ({states})")
        if bio["activity_contexts"]:
            parts.append(f"activity: {', '.join(bio['activity_contexts'])}")
        lines.append(f"- Biometrics: {' · '.join(parts)}")

    # Highlight markers (with cumulative offset inside the group)
    marker_lines: list[str] = []
    offset_cursor = 0.0
    for seg in group.segments:
        for m in seg.markers:
            off = float(m.get("offsetSeconds", 0.0)) + offset_cursor
            marker_lines.append(f"  - ★ `{_format_offset(off)}` (segment `{seg.segment_id[:8]}`)")
        offset_cursor += seg.duration_seconds
    if marker_lines:
        lines.append("- Highlight Markers:")
        lines.extend(marker_lines)

    # Transcript
    lines.append("")
    if cached and cached.get("utterances"):
        langs = ", ".join(cached.get("languages") or [])
        meta = f"engine={cached.get('engine')}"
        if langs:
            meta += f", languages={langs}"
        if cached.get("speaker_count"):
            meta += f", speakers={cached['speaker_count']}"
        lines.append(f"### Transcript ({meta})")
        for utt in cached["utterances"]:
            ts = _format_offset(float(utt.get("start", 0.0)))
            raw_spk = utt.get("speaker")
            spk = speaker_map.get(raw_spk, raw_spk) if raw_spk else None
            prefix = f"[{ts}] **{spk}**: " if spk else f"[{ts}] "
            lines.append(f"{prefix}{utt['text']}")
    elif group.has_existing_transcript:
        lines.append("### Transcript (existing)")
        offset_cursor = 0.0
        for seg in group.segments:
            if seg.existing_transcript:
                lines.append(f"[{_format_offset(offset_cursor)}] {seg.existing_transcript}")
            offset_cursor += seg.duration_seconds
    else:
        lines.append("_No transcript yet. Run `python3 scripts/rfa_context.py transcribe --dir <DIR>`._")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def _parse_dirs(raw_dirs: list[str] | None) -> list[Path]:
    if not raw_dirs:
        return [DEFAULT_INBOX_DIR]
    return [Path(d).expanduser() for d in raw_dirs]


def cmd_check(args: argparse.Namespace) -> int:
    dirs = _parse_dirs(args.dir)
    groups, stats = discover_groups(dirs)
    cache_dir = default_transcript_cache_dir()
    models = resolve_model_paths(Path(args.models_dir) if args.models_dir else None)

    total_segments = sum(len(g.segments) for g in groups)
    cached_groups = sum(1 for g in groups if load_cached_transcript(cache_dir, g.group_id) is not None)
    existing_tx_groups = sum(1 for g in groups if g.has_existing_transcript)
    marker_total = sum(g.marker_count for g in groups)
    bio_groups = sum(1 for g in groups if g.has_biometrics)
    loc_groups = sum(1 for g in groups if g.has_location)

    report = {
        "connected": bool(stats["scanned_dirs"]) and total_segments > 0,
        "scanned_dirs": stats["scanned_dirs"],
        "missing_dirs": stats["missing_dirs"],
        "recording_groups": len(groups),
        "segments": total_segments,
        "duplicates_merged": stats["duplicates_merged"],
        "date_span": {
            "earliest": format_timestamp(groups[0].started_at_ms, groups[0].timezone_id) if groups else None,
            "latest": format_timestamp(groups[-1].ended_at_ms, groups[-1].timezone_id) if groups else None,
        },
        "coverage": {
            "groups_with_existing_transcript": existing_tx_groups,
            "groups_with_cached_local_transcript": cached_groups,
            "highlight_markers": marker_total,
            "groups_with_biometrics": bio_groups,
            "groups_with_location": loc_groups,
        },
        "local_models": models,
    }
    if not report["connected"]:
        report["next_step"] = (
            "Directory is missing or empty. Verify iCloud Mac Companion is running "
            "(writes to ~/.local/share/context-hub-v2/inbox/audio/), or pass --dir pointing to your "
            "Google Drive / WebDAV / local recording folder."
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["connected"] else 2


def cmd_setup(args: argparse.Namespace) -> int:
    status = download_models(
        Path(args.models_dir) if args.models_dir else None,
        include_diarization=not args.no_diarize,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["sensevoice_ready"] else 1


def cmd_list(args: argparse.Namespace) -> int:
    dirs = _parse_dirs(args.dir)
    groups, _ = discover_groups(dirs)
    cache_dir = default_transcript_cache_dir()
    if args.date:
        groups = [g for g in groups if g.local_date_str() == args.date]
    groups = list(reversed(groups))
    if args.limit and args.limit > 0:
        groups = groups[: args.limit]

    items = []
    for g in groups:
        cached = load_cached_transcript(cache_dir, g.group_id)
        items.append(
            {
                "group_id": g.group_id,
                "started_at": format_timestamp(g.started_at_ms, g.timezone_id),
                "duration_seconds": round(g.duration_seconds, 1),
                "segments": len(g.segments),
                "markers": g.marker_count,
                "has_biometrics": g.has_biometrics,
                "has_location": g.has_location,
                "transcribed": bool(cached and cached.get("utterances")) or g.has_existing_transcript,
                "speakers": cached.get("speaker_count") if cached else None,
            }
        )
    print(json.dumps(items, ensure_ascii=False, indent=2))
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    dirs = _parse_dirs(args.dir)
    groups, _ = discover_groups(dirs)
    cache_dir = default_transcript_cache_dir()

    if args.group:
        groups = [g for g in groups if g.group_id.startswith(args.group.lower())]
    if args.date:
        groups = [g for g in groups if g.local_date_str() == args.date]
    if args.limit and args.limit > 0:
        groups = groups[-args.limit :]

    pipeline = LocalAudioPipeline(
        Path(args.models_dir) if args.models_dir else None,
        enable_diarization=not args.no_diarize,
        num_threads=args.threads,
        diarization_threshold=args.diarize_threshold,
    )

    processed = 0
    skipped = 0
    for g in groups:
        if not args.force:
            if load_cached_transcript(cache_dir, g.group_id) is not None:
                skipped += 1
                continue
            if g.has_existing_transcript and not args.include_existing:
                skipped += 1
                continue
        result = pipeline.transcribe_group(g)
        save_cached_transcript(cache_dir, g.group_id, result)
        processed += 1
        print(
            f"Transcribed {g.group_id[:8]} ({len(g.segments)} segs, {result['speaker_count']} speakers, "
            f"{len(result['utterances'])} utterances, {result['elapsed_ms']} ms)",
            file=sys.stderr,
        )

    print(json.dumps({"processed": processed, "skipped": skipped, "cache_dir": str(cache_dir)}, ensure_ascii=False))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    dirs = _parse_dirs(args.dir)
    groups, _ = discover_groups(dirs)
    cache_dir = default_transcript_cache_dir()

    if args.group:
        groups = [g for g in groups if g.group_id.startswith(args.group.lower())]
    if args.date:
        groups = [g for g in groups if g.local_date_str() == args.date]
    if args.limit and args.limit > 0:
        groups = groups[-args.limit :]

    speaker_map: dict[str, str] = {}
    if args.speakers:
        parsed = json.loads(args.speakers)
        if isinstance(parsed, dict):
            speaker_map = {str(k): str(v) for k, v in parsed.items()}

    if args.format == "json":
        out = []
        for g in groups:
            cached = load_cached_transcript(cache_dir, g.group_id)
            out.append(
                {
                    "group_id": g.group_id,
                    "started_at": format_timestamp(g.started_at_ms, g.timezone_id),
                    "duration_seconds": round(g.duration_seconds, 1),
                    "location": next((s.location for s in g.segments if s.location), None),
                    "biometrics": summarize_group_biometrics(g),
                    "markers": [m for s in g.segments for m in s.markers],
                    "transcript": cached,
                }
            )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    rendered = [
        render_group_markdown(g, load_cached_transcript(cache_dir, g.group_id), speaker_map)
        for g in groups
    ]
    print("\n---\n\n".join(rendered))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfa-context",
        description="Zero-API local ingestion, original-language ASR, and speaker diarization for Record for Agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Verify sync folder connectivity and inventory all recordings.")
    p_check.add_argument("--dir", action="append", help="Sync folder path (repeatable; default Mac iCloud inbox).")
    p_check.add_argument("--models-dir", help="Custom model cache directory.")
    p_check.set_defaults(func=cmd_check)

    p_setup = sub.add_parser("setup", help="Download zero-API SenseVoice-Small + Silero-VAD + 3D-Speaker ONNX models.")
    p_setup.add_argument("--models-dir", help="Custom model cache directory.")
    p_setup.add_argument("--no-diarize", action="store_true", help="Skip downloading speaker diarization models.")
    p_setup.set_defaults(func=cmd_setup)

    p_list = sub.add_parser("list", help="List grouped recording sessions.")
    p_list.add_argument("--dir", action="append", help="Sync folder path (repeatable).")
    p_list.add_argument("--date", help="Filter by YYYY-MM-DD.")
    p_list.add_argument("--limit", type=int, default=20, help="Max sessions to return (default 20).")
    p_list.set_defaults(func=cmd_list)

    p_tx = sub.add_parser("transcribe", help="Run local original-language ASR + speaker diarization.")
    p_tx.add_argument("--dir", action="append", help="Sync folder path (repeatable).")
    p_tx.add_argument("--models-dir", help="Custom model cache directory.")
    p_tx.add_argument("--date", help="Filter by YYYY-MM-DD.")
    p_tx.add_argument("--group", help="Filter by recordingGroup.id prefix.")
    p_tx.add_argument("--limit", type=int, default=0, help="Transcribe only the most recent N sessions (0 = all).")
    p_tx.add_argument("--threads", type=int, default=4, help="CPU threads (default 4).")
    p_tx.add_argument("--no-diarize", action="store_true", help="Disable offline speaker diarization.")
    p_tx.add_argument("--diarize-threshold", type=float, default=0.5, help="Speaker clustering threshold (default 0.5; lower = more speakers).")
    p_tx.add_argument("--force", action="store_true", help="Re-transcribe even if cached.")
    p_tx.add_argument("--include-existing", action="store_true", help="Also transcribe segments that already carry on-device text.")
    p_tx.set_defaults(func=cmd_transcribe)

    p_exp = sub.add_parser("export", help="Export rich agent-ready Markdown or JSON.")
    p_exp.add_argument("--dir", action="append", help="Sync folder path (repeatable).")
    p_exp.add_argument("--date", help="Filter by YYYY-MM-DD.")
    p_exp.add_argument("--group", help="Filter by recordingGroup.id prefix.")
    p_exp.add_argument("--limit", type=int, default=5, help="Export the most recent N sessions (default 5).")
    p_exp.add_argument("--speakers", help='JSON speaker rename map, e.g. \'{"Speaker A": "Me"}\'.')
    p_exp.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p_exp.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
