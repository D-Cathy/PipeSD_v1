import unittest
from concurrent.futures import Future
from types import SimpleNamespace

from cloud.core.video_tasks import VideoTaskManager
from cloud.models.video_target import MockVideoTargetBackend
from edge.families.video_speculative.backends import MockVideoDraftBackend
from edge.families.video_speculative.config import VideoSpeculativeConfig
from edge.families.video_speculative.video_edge import VideoSpeculativeEdgeRole
from shared.serialization import unpack_message
from edge.families.video_speculative.strategy import VideoConfidenceStrategy
from pipesd.runtime import Action, CollaborationContext, Result, Task


class InProcessVideoChannel:
    config = SimpleNamespace(server_url="http://cloud")

    def __init__(self):
        self.manager = VideoTaskManager(MockVideoTargetBackend())

    def submit(self, endpoint, data, headers=None, tag=None):
        payload = unpack_message(data)
        future = Future()
        if endpoint.endswith("/video/init"):
            result = self.manager.init_task(payload)
        elif endpoint.endswith("/video/propose"):
            result = self.manager.propose(payload)
        elif endpoint.endswith("/video/exit"):
            result = self.manager.exit_task(payload["task_id"])
        else:
            result = {"error": "unknown endpoint"}
        future.set_result(result)
        return future


class VideoSpeculativeEdgeTests(unittest.TestCase):
    def test_confidence_strategy_returns_framework_decisions(self):
        strategy = VideoConfidenceStrategy(0.9, 0.75)
        task = Task("video-strategy", "video")
        high = strategy.decide(CollaborationContext(task, observations={"average_confidence": 0.95}))
        mid = strategy.decide(CollaborationContext(task, observations={"average_confidence": 0.8}))
        low = strategy.decide(CollaborationContext(task, observations={"average_confidence": 0.2}))
        self.assertEqual((high.action, mid.action, low.action), (
            Action.ACCEPT_LOCAL, Action.SELF_VERIFY, Action.SEND_TO_CLOUD,
        ))

    def test_confidence_routing_and_cloud_override(self):
        draft = MockVideoDraftBackend([0.95, 0.95, 0.2, 0.2, 0.95, 0.95])
        role = VideoSpeculativeEdgeRole(
            draft, InProcessVideoChannel(),
            VideoSpeculativeConfig(chunk_gamma=2, max_new_tokens=8),
        )
        result = role.process_task("video-1", "sample.mp4", "describe")
        self.assertEqual(result["cloud_queries"], 1)
        self.assertEqual(result["tokens"], [10, 11, 13, 14, 15])
        self.assertEqual(result["metrics"]["generated_tokens"], 5)
        self.assertEqual(result["metrics"]["accepted_lengths"], [0])
        self.assertEqual(result["metrics"]["route_counts"]["cloud"], 1)
        self.assertGreater(result["metrics"]["bytes_sent"], 0)
        self.assertGreater(result["metrics"]["bytes_received"], 0)

    def test_public_engine_run_returns_result(self):
        role = VideoSpeculativeEdgeRole(
            MockVideoDraftBackend([0.95, 0.95]), InProcessVideoChannel(),
            VideoSpeculativeConfig(chunk_gamma=2, max_new_tokens=2),
        )
        result = role.run(Task("video-sdk", "video", "sample.mp4", "describe"))
        self.assertIsInstance(result, Result)
        self.assertEqual(result.task_id, "video-sdk")
        self.assertEqual(result.metadata["tokens"], [10, 11])


if __name__ == "__main__":
    unittest.main()
