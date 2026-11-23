"""Unit tests for record-for-agent-skill v0.2 (`scripts/rfa_context.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from scripts import rfa_context as rfa


class RFAContextTests(unittest.TestCase):
    def test_discover_groups_deduplicates_across_cloud_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            icloud_dir = root / "icloud"
            drive_dir = root / "drive"
            dav_dir = root / "webdav"
            icloud_dir.mkdir()
            drive_dir.mkdir()
            dav_dir.mkdir()

            seg1_id = "11111111-1111-1111-1111-111111111111"
            seg2_id = "22222222-2222-2222-2222-222222222222"
            grp_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

            (icloud_dir / f"{seg1_id}.m4a").write_bytes(b"fake-m4a-1")
            (icloud_dir / f"{seg1_id}.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "id": seg1_id,
                        "sessionID": grp_id,
                        "recordingGroupID": grp_id,
                        "sequence": 1,
                        "startedAt": 1763952000000,
                        "endedAt": 1763952060000,
                        "sha256": "abc123",
                        "byteCount": 700000,
                        "timeZoneIdentifier": "Asia/Shanghai",
                        "markers": [{"offsetSeconds": 18.5, "kind": "highlight"}],
                        "biometrics": {
                            "heartRateAvg": 84.0,
                            "heartRateMin": 76.0,
                            "heartRateMax": 96.0,
                            "restingBaselineBPM": 62.0,
                            "arousalScore": 64,
                            "arousalState": "elevated",
                        },
                    }
                ),
                encoding="utf-8",
            )

            (drive_dir / "20251124-0001.manifest.json").write_text(
                json.dumps(
                    {
                        "id": seg1_id,
                        "recordingGroup": {"id": grp_id},
                        "sequence": 1,
                        "startedAt": "2025-11-24T02:40:00Z",
                        "endedAt": "2025-11-24T02:41:00Z",
                        "sha256": "abc123",
                        "byteCount": 700000,
                        "timeZoneIdentifier": "Asia/Shanghai",
                    }
                ),
                encoding="utf-8",
            )
            (drive_dir / "20251124-0001.transcript-deadbeef.txt").write_text(
                "我们下周先发 Beta 版。", encoding="utf-8"
            )

            with zipfile.ZipFile(dav_dir / f"{seg2_id}.ctxaudio.zip", "w") as zf:
                zf.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "id": seg2_id,
                            "recordingGroupID": grp_id,
                            "sequence": 2,
                            "startedAt": 1763952060000,
                            "endedAt": 1763952120000,
                            "sha256": "def456",
                            "byteCount": 680000,
                            "timeZoneIdentifier": "Asia/Shanghai",
                            "location": {
                                "latitude": 31.2304,
                                "longitude": 121.4737,
                                "horizontalAccuracy": 15.0,
                            },
                        }
                    ),
                )
                zf.writestr(f"{seg2_id}.m4a", b"fake-m4a-2")

            groups, stats = rfa.discover_groups([icloud_dir, drive_dir, dav_dir])
            self.assertEqual(len(groups), 1)
            self.assertEqual(stats["duplicates_merged"], 1)
            g = groups[0]
            self.assertEqual([s.sequence for s in g.segments], [1, 2])
            self.assertEqual(g.segments[0].existing_transcript, "我们下周先发 Beta 版。")
            self.assertEqual(g.marker_count, 1)
            self.assertTrue(g.has_biometrics)
            self.assertTrue(g.has_location)

            md = rfa.render_group_markdown(
                g,
                cached={
                    "engine": "sensevoice-small",
                    "languages": ["zh", "en"],
                    "speaker_count": 2,
                    "speakers": {
                        "Speaker A": {"utterance_count": 1, "duration_seconds": 3.5, "suggested_name": "我"},
                        "Speaker B": {"utterance_count": 1, "duration_seconds": 3.0, "suggested_name": None},
                    },
                    "utterances": [
                        {"start": 2.0, "end": 5.5, "event": "Speech", "speaker": "Speaker A", "text": "我们先看下周上线计划。"},
                        {"start": 6.0, "end": 9.0, "event": "Speech", "speaker": "Speaker B", "text": "OK, I will finish tests by Friday."},
                        {"start": 10.0, "end": 12.0, "event": "BGM", "speaker": None, "text": "羊蝎子火锅团购"},
                    ],
                },
                speaker_map={"Speaker B": "Alex"},
            )
            self.assertIn("★ `00:18`", md)
            self.assertIn("14d baseline 62.0 (+22.0 bpm)", md)
            self.assertIn("[00:02] **我**: 我们先看下周上线计划。", md)
            self.assertIn("[00:06] **Alex**: OK, I will finish tests by Friday.", md)
            self.assertIn("[00:10] _[BGM]_ 羊蝎子火锅团购", md)

    def test_average_linkage_clustering_handles_bgm_short_turns_and_llm_num_speakers(self) -> None:
        rng = np.random.default_rng(42)
        base_a = rng.normal(size=64).astype(np.float32)
        base_b = rng.normal(size=64).astype(np.float32)
        # Ensure orthogonal speaker prototypes
        base_b = base_b - (np.dot(base_b, base_a) / np.dot(base_a, base_a)) * base_a

        def noisy(v: np.ndarray, scale: float = 0.15) -> np.ndarray:
            return v + rng.normal(scale=scale, size=v.shape).astype(np.float32)

        utterances = [
            {"start": 0.0, "end": 3.0, "event": "Speech", "text": "老师，我汇报一下项目。"},
            {"start": 3.2, "end": 3.7, "event": "Speech", "text": "嗯"},  # short turn -> backfill
            {"start": 4.0, "end": 7.5, "event": "Speech", "text": "我觉得这个方向不太行。"},
            {"start": 8.0, "end": 12.0, "event": "BGM", "text": "背景广告声音"},  # excluded from clustering
            {"start": 13.0, "end": 16.5, "event": "Speech", "text": "好的，那我记下来。"},
            {"start": 17.0, "end": 20.5, "event": "Speech", "text": "对，不好判断就不推。"},
        ]
        embeddings = [noisy(base_a), noisy(base_b, 0.2), noisy(base_b), None, noisy(base_a), noisy(base_b)]
        vp = {"我": {"embedding": (base_b / np.linalg.norm(base_b)).tolist(), "sample_count": 10}}

        auto = rfa.cluster_utterance_embeddings(utterances, embeddings, voiceprints=vp)
        self.assertEqual(auto["speaker_count"], 2)
        self.assertEqual(auto["utterances"][0]["speaker"], "Speaker A")
        self.assertEqual(auto["utterances"][2]["speaker"], "Speaker B")
        self.assertEqual(auto["utterances"][1]["speaker"], "Speaker B")  # short turn matched cleanly
        self.assertIsNone(auto["utterances"][3]["speaker"])  # BGM excluded
        self.assertEqual(auto["utterances"][4]["speaker"], "Speaker A")
        self.assertEqual(auto["utterances"][5]["speaker"], "Speaker B")
        self.assertEqual(auto["speakers"]["Speaker B"]["suggested_name"], "我")
        self.assertIsNone(auto["speakers"]["Speaker A"]["suggested_name"])

        # LLM-guided single-speaker recluster (num_speakers=1)
        forced_one = rfa.cluster_utterance_embeddings(utterances, embeddings, num_speakers=1, voiceprints=vp)
        self.assertEqual(forced_one["speaker_count"], 1)
        self.assertEqual(forced_one["utterances"][0]["speaker"], "Speaker A")
        self.assertEqual(forced_one["utterances"][2]["speaker"], "Speaker A")

    def test_check_missing_directory_returns_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            groups, stats = rfa.discover_groups([missing])
            self.assertEqual(groups, [])
            self.assertEqual(stats["missing_dirs"], [str(missing)])

    def test_verify_sha256_accepts_exact_match_and_rejects_tampered_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "model.onnx"
            sample.write_bytes(b"record-for-agent-model-bytes")
            digest = rfa.sha256_file(sample)
            rfa.verify_sha256(sample, digest)
            with self.assertRaises(ValueError):
                rfa.verify_sha256(sample, "0" * 64)


if __name__ == "__main__":
    unittest.main()
