import unittest

from cloud.core.video_tasks import VideoTaskManager
from cloud.models.video_target import MockVideoTargetBackend
from shared.protocol import ProtocolError


class VideoTaskManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = VideoTaskManager(MockVideoTargetBackend())
        self.manager.init_task({
            "task_id": "video-1", "prompt": "describe", "model_family": "qwen3_vl",
            "evidence": {"strategy": "mock"}, "generation": {},
        })

    def test_commits_local_tokens_before_cloud_rejection(self):
        response = self.manager.propose({
            "task_id": "video-1", "request_id": "r1", "sequence_no": 0,
            "base_revision": 0, "cache_position": 0,
            "committed_tokens": [10, 11],
            "tokens": [{"token_id": 12, "confidence": 0.2, "topk_ids": [12], "topk_probs": [0.2]}],
            "verification_rule": "js", "js_threshold": 0.4,
        })
        self.assertEqual(response["accepted_count"], 0)
        self.assertEqual(response["override_token"], 13)
        self.assertEqual(response["cache_position"], 3)
        self.assertGreaterEqual(response["cloud_compute_s"], 0.0)

    def test_stale_cache_position_is_rejected(self):
        with self.assertRaises(ProtocolError):
            self.manager.propose({
                "task_id": "video-1", "request_id": "bad", "sequence_no": 0,
                "base_revision": 0, "cache_position": 9, "committed_tokens": [], "tokens": [],
            })


if __name__ == "__main__":
    unittest.main()
