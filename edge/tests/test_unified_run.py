import json
import tempfile
import unittest
from pathlib import Path

from edge.app.run_edge import parse_args as parse_text_args
from edge.app.run_video_edge import load_tasks, parse_args as parse_video_args


class UnifiedRunTests(unittest.TestCase):
    def test_text_legacy_names_map_to_common_names(self):
        args = parse_text_args([
            "--server_url", "http://cloud:8000", "--gamma", "7",
            "--max_generated_tokens", "11", "--draft_model_path", "draft.gguf",
        ])
        self.assertEqual(args.server_url, "http://cloud:8000")
        self.assertEqual(args.chunk_size, 7)
        self.assertEqual(args.max_new_tokens, 11)
        self.assertEqual(args.draft_model_path, "draft.gguf")

    def test_video_config_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "run.json"
            config.write_text(json.dumps({
                "common": {"chunk_size": 4, "max_new_tokens": 16},
                "video": {"max_frames": 8, "video": "sample.mp4"},
            }), encoding="utf-8")
            _, args = parse_video_args([
                "--config", str(config), "--chunk-size", "6",
            ])
            self.assertEqual(args.chunk_size, 6)
            self.assertEqual(args.max_new_tokens, 16)
            self.assertEqual(args.max_frames, 8)
            self.assertEqual(args.video, "sample.mp4")

    def test_video_manifest_loads_multiple_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tasks.jsonl"
            manifest.write_text(
                '{"task_id":"a","video":"a.mp4"}\n'
                '{"task_id":"b","input":"b.mp4","prompt":"describe"}\n',
                encoding="utf-8",
            )
            _, args = parse_video_args(["--input-jsonl", str(manifest)])
            tasks = load_tasks(args)
            self.assertEqual([item["task_id"] for item in tasks], ["a", "b"])
            self.assertEqual(tasks[1]["video"], "b.mp4")


if __name__ == "__main__":
    unittest.main()
