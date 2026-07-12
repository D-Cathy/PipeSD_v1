import unittest

from shared.protocol import InitRequest, ProtocolError, require_protocol
from shared.serialization import pack_message, unpack_message


class ProtocolTests(unittest.TestCase):
    def test_dataclass_round_trip_preserves_version(self):
        payload = unpack_message(pack_message(InitRequest(task_id="task-1", tokens=[1, 2])))
        self.assertEqual(payload["task_id"], "task-1")
        self.assertEqual(payload["tokens"], [1, 2])

    def test_version_mismatch_is_rejected(self):
        with self.assertRaises(ProtocolError):
            require_protocol({"protocol_version": "0.0"})


if __name__ == "__main__":
    unittest.main()
