"""Unit tests for record-for-agent-skill (`scripts/rfa_context.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

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

            # iCloud Companion format (<id>.json + <id>.m4a)
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

            # Duplicate of seg1 in Google Drive format (carries external transcript)
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

            # Segment 2 inside a WebDAV .ctxaudio.zip atomic packet
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
            self.assertIsNotNone(g.segments[0].audio_path)
            self.assertEqual(g.marker_count, 1)
            self.assertTrue(g.has_biometrics)
            self.assertTrue(g.has_location)

            # Export with speaker mapping
            md = rfa.render_group_markdown(
                g,
                cached={
                    "engine": "sensevoice-small",
                    "languages": ["zh", "en"],
                    "speaker_count": 2,
                    "utterances": [
                        {"start": 2.0, "end": 5.5, "speaker": "Speaker A", "text": "我们先看下周上线计划。"},
                        {"start": 6.0, "end": 9.0, "speaker": "Speaker B", "text": "OK, I will finish tests by Friday."},
                    ],
                },
                speaker_map={"Speaker A": "我", "Speaker B": "Alex"},
            )
            self.assertIn("★ `00:18`", md)
            self.assertIn("14d baseline 62.0 (+22.0 bpm)", md)
            self.assertIn("arousal peak 64/100 (elevated)", md)
            self.assertIn("31.23040, 121.47370", md)
            self.assertIn("[00:02] **我**: 我们先看下周上线计划。", md)
            self.assertIn("[00:06] **Alex**: OK, I will finish tests by Friday.", md)

    def test_speaker_assignment_prefers_maximum_overlap(self) -> None:
        timeline = [
            (0.0, 4.0, 0),
            (4.0, 10.0, 1),
            (10.5, 14.0, 0),
        ]
        self.assertEqual(rfa.assign_speaker_to_span(1.0, 3.5, timeline), "Speaker A")
        self.assertEqual(rfa.assign_speaker_to_span(3.5, 8.0, timeline), "Speaker B")
        self.assertEqual(rfa.assign_speaker_to_span(10.6, 13.0, timeline), "Speaker A")
        self.assertEqual(rfa.assign_speaker_to_span(14.5, 15.0, timeline), "Speaker A")  # nearest within 1.5s
        self.assertIsNone(rfa.assign_speaker_to_span(30.0, 31.0, timeline))  # far away

    def test_check_missing_directory_returns_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            groups, stats = rfa.discover_groups([missing])
            self.assertEqual(groups, [])
            self.assertEqual(stats["missing_dirs"], [str(missing)])


if __name__ == "__main__":
    unittest.main()
