import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from concurrent.futures import Future

from core.metrics import MetricsCollector
from core.local_models import MockDraftModel
from families.speculative.speculative_edge import SpeculativeEdgeRole
from families.speculative.strategy import DPStrategy
from shared.protocol import ProtocolError, VerificationResponse


class TextRefactorTests(unittest.TestCase):
    def test_strategy_uses_pending_batch_length(self):
        strategy = DPStrategy(
            SimpleNamespace(gamma=4),
            SimpleNamespace(bandwidth_MBps=2.5, base_latency_c=0.05),
        )
        self.assertFalse(strategy.check_verify_condition(3))
        self.assertTrue(strategy.check_verify_condition(4))

    def test_metrics_writes_official_completion_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = MetricsCollector(directory, "benchmark.json")
            collector.record_custom("output_length", 3)
            collector.save_sample_result("HumanEval/0", " return True", 0, 1.0, "pipesd")
            sample = json.loads(Path(collector.completions_path).read_text(encoding="utf-8"))
            self.assertEqual(sample, {"task_id": "HumanEval/0", "completion": " return True"})
            benchmark = json.loads(Path(collector.saved_path).read_text(encoding="utf-8"))
            self.assertEqual(benchmark[0]["output_length"], 3)

    def test_cloud_error_is_not_reported_as_version_mismatch(self):
        with self.assertRaisesRegex(ProtocolError, "backend exploded"):
            VerificationResponse.from_dict({"error": "backend exploded"})

    def test_runner_verifies_multiple_tokens_and_flushes_tail(self):
        class Channel:
            config = SimpleNamespace(server_url="http://cloud")

            def __init__(self):
                self.sequence = 0
                self.n_past = 1

            def submit(self, endpoint_url, data, headers=None, tag=None):
                from shared.serialization import unpack_message
                future = Future()
                payload = unpack_message(data)
                if endpoint_url.endswith("/init"):
                    future.set_result({"n_past": 1})
                elif endpoint_url.endswith("/exit"):
                    future.set_result({"status": "exited"})
                else:
                    count = len(payload["tokens"])
                    self.n_past += count + 1
                    self.sequence += 1
                    future.set_result({
                        "protocol_version": "1.0", "task_id": payload["task_id"],
                        "request_id": payload["request_id"], "sequence_no": payload["sequence_no"],
                        "revision": self.sequence, "n_accepted": count,
                        "n_speculative": count, "final_token": 99, "n_past": self.n_past,
                    })
                return future

            def drain_tag(self, tag):
                return []

        class Strategy:
            def check_verify_condition(self, pending_length):
                return pending_length >= 3

        with tempfile.TemporaryDirectory() as directory:
            collector = MetricsCollector(directory, "benchmark.json")
            role = SpeculativeEdgeRole(
                MockDraftModel(), Channel(), Strategy(), collector, SimpleNamespace(),
                SimpleNamespace(max_generated_tokens=8, algorithm="pipesd"),
            )
            role.process_task("HumanEval/0", "def f():")
            self.assertEqual(collector.verify_spec_lengths, [3, 3])
            self.assertEqual(collector.current_metrics["output_length"], 8)


if __name__ == "__main__":
    unittest.main()
