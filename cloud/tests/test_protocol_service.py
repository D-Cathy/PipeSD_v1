import unittest

from cloud.core.tasks import TaskManager
from cloud.models import MockTargetBackend
from shared.protocol import InitRequest, ProposalRequest, ProtocolError


class TaskManagerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manager = TaskManager(MockTargetBackend())
        self.manager.init_task(InitRequest(task_id="a", tokens=[10, 11]).to_dict())

    def test_n_past_revision_and_sequence_advance_together(self):
        response = self.manager.propose(ProposalRequest(
            task_id="a", request_id="r1", sequence_no=0, base_revision=0,
            n_past=2, tokens=[12, 13], probs=[[1.0], [1.0]],
        ).to_dict())
        self.assertEqual(response["n_past"], 5)
        self.assertEqual(response["revision"], 1)
        self.assertEqual(response["sequence_no"], 0)

    def test_duplicate_request_is_idempotent(self):
        request = ProposalRequest(
            task_id="a", request_id="same", sequence_no=0, base_revision=0,
            n_past=2, tokens=[12], probs=[[1.0]],
        ).to_dict()
        self.assertEqual(self.manager.propose(request), self.manager.propose(request))

    def test_stale_n_past_is_rejected(self):
        request = ProposalRequest(
            task_id="a", request_id="bad", sequence_no=0, base_revision=0,
            n_past=1, tokens=[12], probs=[[1.0]],
        ).to_dict()
        with self.assertRaises(ProtocolError):
            self.manager.propose(request)

    def test_exit_cleans_task(self):
        self.manager.exit_task("a")
        self.assertNotIn("a", self.manager.tasks)


if __name__ == "__main__":
    unittest.main()
