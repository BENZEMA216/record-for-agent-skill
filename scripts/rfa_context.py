#!/usr/bin/env python3
"""Zero-API local ingestion, original-language ASR, and Dual-Factor (Voiceprint + LLM Context)
speaker diarization for Record for Agent.

Supports:
- iCloud Mac Companion inbox (`~/.local/share/context-hub-v2/inbox/audio/`: `<id>.m4a` + `<id>.json`)
- Google Drive synced folders (`<prefix>.manifest.json` + `<prefix>.m4a`/`.caf` + `.transcript-*`)
- WebDAV atomic packets (`*.ctxaudio.zip` containing `manifest.json` + `.m4a`)

Never modifies, renames, or deletes source files. Cached transcripts, voiceprint embeddings,
and ONNX models live strictly under `~/.cache/record-for-agent/`.
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
SPEAKER_EMBED_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
SPEAKER_EMBED_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    f"{SPEAKER_EMBED_NAME}"
)

NON_SPEECH_EVENTS = {"BGM", "Music", "Applause", "Noise"}
DEFAULT_SIM_THRESHOLD = 0.35
MIN_CLUSTER_DURATION_S = 1.2
VOICEPRINT_MATCH_THRESHOLD = 0.38
VOICEPRINT_MATCH_MARGIN = 0.08


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
        if num > 1e11:
            return int(round(num))
        if num > 1e9:
            return int(round(num * 1000))
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


def default_voiceprints_path() -> Path:
    if env := os.environ.get("RFA_VOICEPRINTS_PATH"):
        return Path(env).expanduser()
    return DEFAULT_CACHE_ROOT / "voiceprints.json"


def resolve_model_paths(models_dir: Path | None = None) -> dict[str, Any]:
    primary = (models_dir or default_models_dir()).expanduser()
    search_roots = [primary]
    if CONTEXT_HUB_MODELS_DIR != primary and CONTEXT_HUB_MODELS_DIR.is_dir():
        search_roots.append(CONTEXT_HUB_MODELS_DIR)

    sv_model: Path | None = None
    sv_tokens: Path | None = None
    vad_path: Path | None = None
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
        "diarization_ready": embed_model is not None,
        "embedding_model": str(embed_model) if embed_model else None,
    }


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "record-for-agent-skill/0.2"})
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
        _download_file(SPEAKER_EMBED_URL, root / SPEAKER_EMBED_NAME)

    return resolve_model_paths(root)


# ---------------------------------------------------------------------------
# Voiceprint Clustering & Cross-Day Voiceprint Registry (Pure NumPy)
# ---------------------------------------------------------------------------


def speaker_index_to_label(index: int) -> str:
    if 0 <= index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def _l2_normalize(vec: Any) -> Any:
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def load_voiceprints(path: Path | None = None) -> dict[str, dict[str, Any]]:
    vp_path = (path or default_voiceprints_path()).expanduser()
    if not vp_path.is_file():
        return {}
    try:
        data = json.loads(vp_path.read_text(encoding="utf-8"))
        speakers = data.get("speakers") if isinstance(data, dict) else None
        return speakers if isinstance(speakers, dict) else {}
    except Exception:
        return {}


def save_voiceprints(speakers: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    vp_path = (path or default_voiceprints_path()).expanduser()
    vp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = vp_path.with_suffix(".json.tmp")
    payload = {"schema_version": 1, "speakers": speakers}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(vp_path)
    return vp_path


def cluster_utterance_embeddings(
    utterances: list[dict[str, Any]],
    embeddings: list[Any],
    *,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    num_speakers: int | None = None,
    min_cluster_duration: float = MIN_CLUSTER_DURATION_S,
    voiceprints: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dual-Factor-friendly Average-Linkage Cosine Clustering on 3D-Speaker embeddings.

    1. Excludes `<|BGM|>` / non-speech events and short `<1.2s` interjections from cluster
       formation so background ads and brief "嗯/对" replies never fragment speakers.
    2. Clusters clean speech embeddings via exact average-linkage cosine similarity
       (either automatically down to `sim_threshold`, or constrained to LLM-estimated `num_speakers`).
    3. Backfills short turns by nearest cluster centroid (cos_sim >= 0.25) or adjacent turn.
    4. Emits rich acoustic evidence (`cluster_similarity_matrix`, `voiceprint_matches`,
       per-utterance `sim_to_prev` & `speaker_confidence`) so an LLM combines voiceprint +
       conversational semantics rather than blindly trusting acoustics alone.
    """
    import numpy as np

    n = len(utterances)
    if n == 0:
        return {
            "speaker_count": 0,
            "diarized": False,
            "sim_threshold": sim_threshold,
            "forced_num_speakers": num_speakers,
            "speakers": {},
            "cluster_similarity_matrix": {},
            "utterances": [],
        }

    norm_embs: list[Any] = [
        _l2_normalize(embeddings[i]) if i < len(embeddings) and embeddings[i] is not None else None
        for i in range(n)
    ]

    # Select clean speech utterances for cluster formation
    seed_indices: list[int] = []
    for i, utt in enumerate(utterances):
        dur = float(utt.get("end", 0.0)) - float(utt.get("start", 0.0))
        event = str(utt.get("event") or "Speech")
        if norm_embs[i] is not None and dur >= min_cluster_duration and event not in NON_SPEECH_EVENTS:
            seed_indices.append(i)

    # Relax duration threshold if entire recording is made of brief turns
    if not seed_indices:
        for i, utt in enumerate(utterances):
            event = str(utt.get("event") or "Speech")
            if norm_embs[i] is not None and event not in NON_SPEECH_EVENTS:
                seed_indices.append(i)

    if not seed_indices:
        cleaned = [dict(u, speaker=None, speaker_confidence=None, sim_to_prev=None) for u in utterances]
        return {
            "speaker_count": 0,
            "diarized": False,
            "sim_threshold": sim_threshold,
            "forced_num_speakers": num_speakers,
            "speakers": {},
            "cluster_similarity_matrix": {},
            "utterances": cleaned,
        }

    # Build pairwise cosine similarity matrix across seed utterances
    m = len(seed_indices)
    mat = np.stack([norm_embs[i] for i in seed_indices], axis=0)
    sim_matrix = mat @ mat.T  # shape (m, m), values in [-1, 1]

    # Average-linkage agglomerative clustering
    clusters: list[list[int]] = [[k] for k in range(m)]
    target_k = max(1, int(num_speakers)) if num_speakers is not None and num_speakers > 0 else None

    if target_k == 1:
        clusters = [list(range(m))]
    else:
        while len(clusters) > 1:
            best_pair: tuple[int, int] | None = None
            best_sim = -2.0
            c_count = len(clusters)
            for a in range(c_count):
                idx_a = clusters[a]
                for b in range(a + 1, c_count):
                    idx_b = clusters[b]
                    avg_sim = float(sim_matrix[np.ix_(idx_a, idx_b)].mean())
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (a, b)
            if best_pair is None:
                break
            if target_k is not None:
                if len(clusters) <= target_k:
                    break
            else:
                if best_sim < sim_threshold:
                    break
            a, b = best_pair
            clusters[a] = clusters[a] + clusters[b]
            clusters.pop(b)

        # Secondary centroid-linkage pass in auto mode:
        # Averaging noisy phone utterances into centroids cancels channel noise, so merge any two
        # clusters whose L2-normalized centroids still have cosine similarity >= sim_threshold + 0.05
        # (or absorb tiny <=2-utterance fragments into a major cluster if centroid sim >= 0.28).
        if target_k is None and len(clusters) > 1:
            def _centroid(member_ks: list[int]) -> Any:
                return _l2_normalize(np.stack([norm_embs[seed_indices[k]] for k in member_ks], axis=0).mean(axis=0))

            changed = True
            while changed and len(clusters) > 1:
                changed = False
                c_vecs = [_centroid(c) for c in clusters]
                best_pair = None
                best_sim = -2.0
                for a in range(len(clusters)):
                    for b in range(a + 1, len(clusters)):
                        if c_vecs[a] is None or c_vecs[b] is None:
                            continue
                        s = float(np.dot(c_vecs[a], c_vecs[b]))
                        is_tiny_vs_major = (len(clusters[a]) <= 2 and len(clusters[b]) >= 4) or (
                            len(clusters[b]) <= 2 and len(clusters[a]) >= 4
                        )
                        cutoff = max(0.26, sim_threshold - 0.07) if is_tiny_vs_major else (sim_threshold + 0.05)
                        if s >= cutoff and s > best_sim:
                            best_sim = s
                            best_pair = (a, b)
                if best_pair is not None:
                    a, b = best_pair
                    clusters[a] = clusters[a] + clusters[b]
                    clusters.pop(b)
                    changed = True

    # Sort clusters by earliest appearance in the session
    clusters.sort(key=lambda c: min(seed_indices[k] for k in c))

    # Compute L2-normalized cluster centroids
    centroids: list[Any] = []
    seed_cluster_map: dict[int, int] = {}
    for cid, member_ks in enumerate(clusters):
        vecs = np.stack([norm_embs[seed_indices[k]] for k in member_ks], axis=0)
        c_vec = _l2_normalize(vecs.mean(axis=0))
        centroids.append(c_vec)
        for k in member_ks:
            seed_cluster_map[seed_indices[k]] = cid

    # Assign all utterances (seed turns keep their cluster; short turns backfill by centroid or neighbor)
    assigned_cid: list[int | None] = [None] * n
    assigned_conf: list[float | None] = [None] * n
    for i, utt in enumerate(utterances):
        event = str(utt.get("event") or "Speech")
        if event in NON_SPEECH_EVENTS:
            continue
        if i in seed_cluster_map:
            cid = seed_cluster_map[i]
            assigned_cid[i] = cid
            if norm_embs[i] is not None and centroids[cid] is not None:
                assigned_conf[i] = round(float(np.dot(norm_embs[i], centroids[cid])), 3)
        elif norm_embs[i] is not None:
            sims = [float(np.dot(norm_embs[i], c)) if c is not None else -1.0 for c in centroids]
            best_cid = int(np.argmax(sims))
            best_sim = sims[best_cid]
            if best_sim >= 0.25 or len(centroids) == 1:
                assigned_cid[i] = best_cid
                assigned_conf[i] = round(best_sim, 3)

    # Neighbor fallback for ultra-short speech turns without reliable embedding match
    for i, utt in enumerate(utterances):
        event = str(utt.get("event") or "Speech")
        if event in NON_SPEECH_EVENTS or assigned_cid[i] is not None:
            continue
        nearest_cid = None
        nearest_gap = 4.0
        start_i = float(utt.get("start", 0.0))
        for j in range(n):
            if assigned_cid[j] is None:
                continue
            gap = abs(float(utterances[j].get("start", 0.0)) - start_i)
            if gap < nearest_gap:
                nearest_gap = gap
                nearest_cid = assigned_cid[j]
        if nearest_cid is not None:
            assigned_cid[i] = nearest_cid
            assigned_conf[i] = None

    # Pairwise cluster similarity matrix (for LLM cross-checking)
    cluster_sim: dict[str, dict[str, float]] = {}
    for a in range(len(centroids)):
        la = speaker_index_to_label(a)
        cluster_sim[la] = {}
        for b in range(len(centroids)):
            lb = speaker_index_to_label(b)
            if centroids[a] is not None and centroids[b] is not None:
                cluster_sim[la][lb] = round(float(np.dot(centroids[a], centroids[b])), 3)

    # Match cluster centroids against enrolled cross-day voiceprints
    vp_db = voiceprints if voiceprints is not None else load_voiceprints()
    vp_vecs: dict[str, Any] = {}
    for name, info in vp_db.items():
        if isinstance(info, dict) and isinstance(info.get("embedding"), list):
            v = _l2_normalize(info["embedding"])
            if v is not None:
                vp_vecs[name] = v

    speakers_summary: dict[str, dict[str, Any]] = {}
    for cid, c_vec in enumerate(centroids):
        label = speaker_index_to_label(cid)
        member_indices = [i for i in range(n) if assigned_cid[i] == cid]
        total_dur = sum(float(utterances[i].get("end", 0.0)) - float(utterances[i].get("start", 0.0)) for i in member_indices)
        # Representative quote: longest clean utterance from this cluster
        rep_quote = None
        if member_indices:
            best_i = max(member_indices, key=lambda i: len(str(utterances[i].get("text") or "")))
            rep_quote = {
                "start": utterances[best_i].get("start"),
                "text": str(utterances[best_i].get("text") or "")[:120],
            }

        matches: list[dict[str, Any]] = []
        suggested_name = None
        if c_vec is not None and vp_vecs:
            scored = sorted(
                ((name, float(np.dot(c_vec, v))) for name, v in vp_vecs.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            matches = [{"name": name, "similarity": round(score, 3)} for name, score in scored[:3]]
            top_name, top_score = scored[0]
            second_score = scored[1][1] if len(scored) > 1 else -1.0
            if top_score >= VOICEPRINT_MATCH_THRESHOLD and (top_score - second_score) >= VOICEPRINT_MATCH_MARGIN:
                suggested_name = top_name

        speakers_summary[label] = {
            "utterance_count": len(member_indices),
            "duration_seconds": round(total_dur, 1),
            "suggested_name": suggested_name,
            "voiceprint_matches": matches,
            "representative_quote": rep_quote,
        }

    # Attach final labels + consecutive similarity cue (`sim_to_prev`) to each utterance
    enriched: list[dict[str, Any]] = []
    prev_emb = None
    for i, utt in enumerate(utterances):
        item = dict(utt)
        cid = assigned_cid[i]
        event = str(utt.get("event") or "Speech")
        if event in NON_SPEECH_EVENTS:
            item["speaker"] = None
            item["speaker_confidence"] = None
            item["sim_to_prev"] = None
        else:
            item["speaker"] = speaker_index_to_label(cid) if cid is not None else None
            item["speaker_confidence"] = assigned_conf[i]
            if norm_embs[i] is not None and prev_emb is not None:
                item["sim_to_prev"] = round(float(np.dot(norm_embs[i], prev_emb)), 3)
            else:
                item["sim_to_prev"] = None
            if norm_embs[i] is not None:
                prev_emb = norm_embs[i]
        enriched.append(item)

    return {
        "speaker_count": len(centroids),
        "diarized": True,
        "sim_threshold": sim_threshold,
        "forced_num_speakers": target_k,
        "speakers": speakers_summary,
        "cluster_similarity_matrix": cluster_sim,
        "utterances": enriched,
    }


# ---------------------------------------------------------------------------
# Audio Decode & Original-Language ASR + 3D-Speaker Embedding Pipeline
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


class LocalAudioPipeline:
    """Runs SenseVoice-Small (auto spoken language + event tags) + Silero-VAD + 3D-Speaker embeddings."""

    def __init__(
        self,
        models_dir: Path | None = None,
        *,
        enable_diarization: bool = True,
        num_threads: int = 4,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        num_speakers: int | None = None,
    ):
        self.models_dir = models_dir
        self.enable_diarization = enable_diarization
        self.num_threads = max(1, num_threads)
        self.sim_threshold = sim_threshold
        self.num_speakers = num_speakers
        self._recognizer: Any = None
        self._vad_config: Any = None
        self._embedder: Any = None

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
            cfg.silero_vad.min_silence_duration = 0.35
            cfg.silero_vad.min_speech_duration = 0.25
            cfg.silero_vad.max_speech_duration = 12.0
            cfg.silero_vad.window_size = 512
            cfg.sample_rate = 16000
            cfg.num_threads = 1
            self._vad_config = cfg

        if self.enable_diarization and status["diarization_ready"]:
            emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=status["embedding_model"],
                num_threads=self.num_threads,
                provider="cpu",
            )
            self._embedder = sherpa_onnx.SpeakerEmbeddingExtractor(emb_cfg)

    def _vad_chunks(self, samples: Any, sr: int) -> list[tuple[float, float, Any]]:
        import numpy as np

        if self._vad_config is None:
            win = 12 * sr
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

    def _extract_embedding(self, samples: Any) -> Any:
        if self._embedder is None or len(samples) < 6400:  # < 0.4s too noisy
            return None
        stream = self._embedder.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        if not self._embedder.is_ready(stream):
            return None
        return _l2_normalize(self._embedder.compute(stream))

    def transcribe_group(self, group: RecordingGroup) -> tuple[dict[str, Any], list[Any]]:
        import numpy as np

        t0 = time.perf_counter()
        self._ensure_loaded()

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
            return (
                {
                    "schema_version": 2,
                    "group_id": group.group_id,
                    "engine": "sensevoice-small",
                    "diarized": False,
                    "speaker_count": 0,
                    "languages": [],
                    "speakers": {},
                    "cluster_similarity_matrix": {},
                    "utterances": [],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                },
                [],
            )

        full_pcm = np.concatenate(pcm_parts)
        peak = float(max(abs(full_pcm.min()), abs(full_pcm.max()))) if len(full_pcm) else 0.0
        if len(full_pcm) < 3200 or peak < 1e-4:
            return (
                {
                    "schema_version": 2,
                    "group_id": group.group_id,
                    "engine": "sensevoice-small",
                    "diarized": False,
                    "speaker_count": 0,
                    "languages": [],
                    "speakers": {},
                    "cluster_similarity_matrix": {},
                    "utterances": [],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                },
                [],
            )

        raw_utterances: list[dict[str, Any]] = []
        raw_embeddings: list[Any] = []
        languages: set[str] = set()

        for start_s, end_s, chunk in self._vad_chunks(full_pcm, 16000):
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, chunk)
            self._recognizer.decode_stream(stream)
            text = (stream.result.text or "").strip()
            if not text:
                continue
            lang = (getattr(stream.result, "lang", "") or "").strip().strip("<|>")
            event = (getattr(stream.result, "event", "") or "").strip().strip("<|>") or "Speech"
            emotion = (getattr(stream.result, "emotion", "") or "").strip().strip("<|>")
            if lang and event not in NON_SPEECH_EVENTS:
                languages.add(lang)

            mid = 0.5 * (start_s + end_s)
            seg_id = group.segments[0].segment_id
            for seg, seg_start, seg_end in segment_spans:
                if seg_start <= mid <= seg_end:
                    seg_id = seg.segment_id
                    break

            emb = self._extract_embedding(chunk) if event not in NON_SPEECH_EVENTS else None
            raw_utterances.append(
                {
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "lang": lang or None,
                    "event": event,
                    "emotion": emotion if emotion and emotion != "EMO_UNKNOWN" else None,
                    "segment_id": seg_id,
                    "text": text,
                }
            )
            raw_embeddings.append(emb)

        if self._embedder is not None and raw_utterances:
            clustered = cluster_utterance_embeddings(
                raw_utterances,
                raw_embeddings,
                sim_threshold=self.sim_threshold,
                num_speakers=self.num_speakers,
            )
            payload = {
                "schema_version": 2,
                "group_id": group.group_id,
                "engine": "sensevoice-small",
                "diarized": clustered["diarized"],
                "speaker_count": clustered["speaker_count"],
                "sim_threshold": clustered["sim_threshold"],
                "forced_num_speakers": clustered["forced_num_speakers"],
                "languages": sorted(languages),
                "speakers": clustered["speakers"],
                "cluster_similarity_matrix": clustered["cluster_similarity_matrix"],
                "utterances": clustered["utterances"],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            }
        else:
            payload = {
                "schema_version": 2,
                "group_id": group.group_id,
                "engine": "sensevoice-small",
                "diarized": False,
                "speaker_count": 1 if raw_utterances else 0,
                "languages": sorted(languages),
                "speakers": {},
                "cluster_similarity_matrix": {},
                "utterances": [dict(u, speaker=None) for u in raw_utterances],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            }
        return payload, raw_embeddings


# ---------------------------------------------------------------------------
# Cache (Transcript JSON + Compact Float16 Embeddings Sidecar) & Rendering
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


def save_cached_transcript(
    cache_dir: Path,
    group_id: str,
    payload: dict[str, Any],
    embeddings: list[Any] | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{group_id}.json"
    tmp = cache_dir / f".tmp-{group_id}.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)

    if embeddings is not None:
        import numpy as np

        emb_dest = cache_dir / f"{group_id}.embeddings.npz"
        emb_tmp = cache_dir / f".tmp-{group_id}.embeddings.npz"
        dim = next((len(e) for e in embeddings if e is not None), 0)
        if dim > 0:
            mat = np.zeros((len(embeddings), dim), dtype=np.float16)
            valid = np.zeros((len(embeddings),), dtype=np.bool_)
            for i, e in enumerate(embeddings):
                if e is not None:
                    mat[i] = np.asarray(e, dtype=np.float16)
                    valid[i] = True
            with open(emb_tmp, "wb") as f:
                np.savez_compressed(f, embeddings=mat, valid=valid)
            emb_tmp.replace(emb_dest)
    return dest


def load_cached_embeddings(cache_dir: Path, group_id: str) -> list[Any] | None:
    import numpy as np

    emb_path = cache_dir / f"{group_id}.embeddings.npz"
    if not emb_path.is_file():
        return None
    try:
        with np.load(emb_path) as data:
            mat = data["embeddings"].astype(np.float32)
            valid = data["valid"]
            return [mat[i] if bool(valid[i]) else None for i in range(len(valid))]
    except Exception:
        return None


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

    # Highlight markers (cumulative offset inside group)
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

    # Voiceprint cluster summary for LLM dual-factor verification
    effective_map = dict(speaker_map)
    if cached and isinstance(cached.get("speakers"), dict) and cached["speakers"]:
        lines.append("- Acoustic Voiceprint Clusters (verify jointly with dialogue context):")
        for label, info in cached["speakers"].items():
            sug = info.get("suggested_name")
            if sug and label not in effective_map:
                effective_map[label] = sug
            match_str = ""
            if info.get("voiceprint_matches"):
                top = info["voiceprint_matches"][0]
                match_str = f", voiceprint match `{top['name']}` ({top['similarity']:+.3f})"
            lines.append(
                f"  - `{label}`: {info.get('utterance_count', 0)} turns / {info.get('duration_seconds', 0)}s{match_str}"
            )

    lines.append("")
    if cached and cached.get("utterances"):
        langs = ", ".join(cached.get("languages") or [])
        meta = f"engine={cached.get('engine')}"
        if langs:
            meta += f", languages={langs}"
        if cached.get("speaker_count"):
            meta += f", acoustic_speakers={cached['speaker_count']}"
        lines.append(f"### Transcript ({meta})")
        for utt in cached["utterances"]:
            ts = _format_offset(float(utt.get("start", 0.0)))
            event = str(utt.get("event") or "Speech")
            if event in NON_SPEECH_EVENTS:
                lines.append(f"[{ts}] _[{event}]_ {utt['text']}")
                continue
            raw_spk = utt.get("speaker")
            spk = effective_map.get(raw_spk, raw_spk) if raw_spk else None
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


def _resolve_cached_group_id(cache_dir: Path, prefix: str) -> str | None:
    target = prefix.strip().lower()
    if (cache_dir / f"{target}.json").is_file():
        return target
    matches = sorted(
        p.stem for p in cache_dir.glob(f"{target}*.json") if not p.name.startswith(".")
    )
    return matches[0] if len(matches) == 1 else None


def cmd_check(args: argparse.Namespace) -> int:
    dirs = _parse_dirs(args.dir)
    groups, stats = discover_groups(dirs)
    cache_dir = default_transcript_cache_dir()
    models = resolve_model_paths(Path(args.models_dir) if args.models_dir else None)
    vps = load_voiceprints()

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
        "enrolled_voiceprints": sorted(vps.keys()),
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
        sim_threshold=args.sim_threshold,
        num_speakers=args.num_speakers,
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
        result, embeddings = pipeline.transcribe_group(g)
        save_cached_transcript(cache_dir, g.group_id, result, embeddings=embeddings)
        processed += 1
        print(
            f"Transcribed {g.group_id[:8]} ({len(g.segments)} segs, {result['speaker_count']} speakers, "
            f"{len(result['utterances'])} utterances, {result['elapsed_ms']} ms)",
            file=sys.stderr,
        )

    print(json.dumps({"processed": processed, "skipped": skipped, "cache_dir": str(cache_dir)}, ensure_ascii=False))
    return 0


def cmd_recluster(args: argparse.Namespace) -> int:
    """Instant sub-second reclustering on cached embeddings using LLM-estimated `--num-speakers K`."""
    cache_dir = default_transcript_cache_dir()
    group_id = _resolve_cached_group_id(cache_dir, args.group)
    if not group_id:
        print(json.dumps({"error": f"No unique cached transcript found for group prefix '{args.group}'."}))
        return 1
    cached = load_cached_transcript(cache_dir, group_id)
    embeddings = load_cached_embeddings(cache_dir, group_id)
    if not cached or embeddings is None:
        print(json.dumps({"error": f"No cached embeddings for '{group_id}'. Run `transcribe --force --group {group_id[:8]}` first."}))
        return 1

    clustered = cluster_utterance_embeddings(
        cached.get("utterances") or [],
        embeddings,
        sim_threshold=args.sim_threshold,
        num_speakers=args.num_speakers,
    )
    updated = dict(cached)
    updated["schema_version"] = 2
    updated["diarized"] = clustered["diarized"]
    updated["speaker_count"] = clustered["speaker_count"]
    updated["sim_threshold"] = clustered["sim_threshold"]
    updated["forced_num_speakers"] = clustered["forced_num_speakers"]
    updated["speakers"] = clustered["speakers"]
    updated["cluster_similarity_matrix"] = clustered["cluster_similarity_matrix"]
    updated["utterances"] = clustered["utterances"]
    save_cached_transcript(cache_dir, group_id, updated)
    print(
        json.dumps(
            {
                "group_id": group_id,
                "speaker_count": updated["speaker_count"],
                "speakers": updated["speakers"],
                "cluster_similarity_matrix": updated["cluster_similarity_matrix"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    """Save a confirmed speaker's voiceprint centroid into `~/.cache/record-for-agent/voiceprints.json`."""
    import numpy as np

    cache_dir = default_transcript_cache_dir()
    group_id = _resolve_cached_group_id(cache_dir, args.group)
    if not group_id:
        print(json.dumps({"error": f"No unique cached transcript found for group prefix '{args.group}'."}))
        return 1
    cached = load_cached_transcript(cache_dir, group_id)
    embeddings = load_cached_embeddings(cache_dir, group_id)
    if not cached or embeddings is None:
        print(json.dumps({"error": f"No cached embeddings for '{group_id}'."}))
        return 1

    target_label = args.speaker.strip()
    utterances = cached.get("utterances") or []
    vecs = []
    for i, utt in enumerate(utterances):
        if utt.get("speaker") == target_label and i < len(embeddings) and embeddings[i] is not None:
            dur = float(utt.get("end", 0.0)) - float(utt.get("start", 0.0))
            if dur >= MIN_CLUSTER_DURATION_S:
                v = _l2_normalize(embeddings[i])
                if v is not None:
                    vecs.append(v)
    if not vecs:
        print(json.dumps({"error": f"No valid speech embeddings found for '{target_label}' in group '{group_id[:8]}'."}))
        return 1

    new_centroid = _l2_normalize(np.stack(vecs, axis=0).mean(axis=0))
    db = load_voiceprints()
    existing = db.get(args.name)
    if existing and isinstance(existing.get("embedding"), list) and not args.replace:
        old_vec = _l2_normalize(existing["embedding"])
        old_n = int(existing.get("sample_count", 1))
        combined = _l2_normalize(old_vec * old_n + new_centroid * len(vecs))
        total_n = old_n + len(vecs)
    else:
        combined = new_centroid
        total_n = len(vecs)

    db[args.name] = {
        "embedding": [round(float(x), 6) for x in combined.tolist()],
        "sample_count": total_n,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_group_id": group_id,
    }
    vp_path = save_voiceprints(db)
    print(
        json.dumps(
            {
                "enrolled": args.name,
                "from_group": group_id,
                "from_speaker": target_label,
                "samples_added": len(vecs),
                "total_samples": total_n,
                "voiceprints_path": str(vp_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_voiceprints(args: argparse.Namespace) -> int:
    db = load_voiceprints()
    summary = {
        name: {
            "sample_count": info.get("sample_count"),
            "updated_at": info.get("updated_at"),
            "last_group_id": info.get("last_group_id"),
        }
        for name, info in db.items()
    }
    print(json.dumps({"voiceprints_path": str(default_voiceprints_path()), "speakers": summary}, ensure_ascii=False, indent=2))
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
        description="Zero-API local ingestion, original-language ASR, and Dual-Factor speaker diarization for Record for Agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Verify sync folder connectivity and inventory all recordings.")
    p_check.add_argument("--dir", action="append", help="Sync folder path (repeatable; default Mac iCloud inbox).")
    p_check.add_argument("--models-dir", help="Custom model cache directory.")
    p_check.set_defaults(func=cmd_check)

    p_setup = sub.add_parser("setup", help="Download zero-API SenseVoice-Small + Silero-VAD + 3D-Speaker ONNX models.")
    p_setup.add_argument("--models-dir", help="Custom model cache directory.")
    p_setup.add_argument("--no-diarize", action="store_true", help="Skip downloading 3D-Speaker voiceprint model.")
    p_setup.set_defaults(func=cmd_setup)

    p_list = sub.add_parser("list", help="List grouped recording sessions.")
    p_list.add_argument("--dir", action="append", help="Sync folder path (repeatable).")
    p_list.add_argument("--date", help="Filter by YYYY-MM-DD.")
    p_list.add_argument("--limit", type=int, default=20, help="Max sessions to return (default 20).")
    p_list.set_defaults(func=cmd_list)

    p_tx = sub.add_parser("transcribe", help="Run local original-language ASR + 3D-Speaker voiceprint clustering.")
    p_tx.add_argument("--dir", action="append", help="Sync folder path (repeatable).")
    p_tx.add_argument("--models-dir", help="Custom model cache directory.")
    p_tx.add_argument("--date", help="Filter by YYYY-MM-DD.")
    p_tx.add_argument("--group", help="Filter by recordingGroup.id prefix.")
    p_tx.add_argument("--limit", type=int, default=0, help="Transcribe only the most recent N sessions (0 = all).")
    p_tx.add_argument("--threads", type=int, default=4, help="CPU threads (default 4).")
    p_tx.add_argument("--no-diarize", action="store_true", help="Disable voiceprint embedding extraction.")
    p_tx.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD, help="Cosine similarity merge threshold (default 0.35).")
    p_tx.add_argument("--num-speakers", type=int, default=None, help="Optional LLM-guided speaker count constraint.")
    p_tx.add_argument("--force", action="store_true", help="Re-transcribe even if cached.")
    p_tx.add_argument("--include-existing", action="store_true", help="Also transcribe segments that already carry on-device text.")
    p_tx.set_defaults(func=cmd_transcribe)

    p_re = sub.add_parser("recluster", help="Instantly re-cluster cached voiceprint embeddings with LLM-guided speaker count.")
    p_re.add_argument("--group", required=True, help="recordingGroup.id prefix.")
    p_re.add_argument("--num-speakers", type=int, default=None, help="LLM-estimated number of speakers (e.g. 1, 2, 3).")
    p_re.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD, help="Cosine similarity merge threshold (default 0.35).")
    p_re.set_defaults(func=cmd_recluster)

    p_en = sub.add_parser("enroll", help="Save a confirmed speaker's voiceprint locally for cross-day auto-recognition.")
    p_en.add_argument("--group", required=True, help="recordingGroup.id prefix.")
    p_en.add_argument("--speaker", required=True, help='Speaker label from transcript (e.g. "Speaker A").')
    p_en.add_argument("--name", required=True, help='Confirmed display name (e.g. "我" or "Alex").')
    p_en.add_argument("--replace", action="store_true", help="Replace existing voiceprint instead of averaging.")
    p_en.set_defaults(func=cmd_enroll)

    p_vp = sub.add_parser("voiceprints", help="List locally enrolled cross-day speaker voiceprints.")
    p_vp.set_defaults(func=cmd_voiceprints)

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
