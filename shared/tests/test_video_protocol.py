import unittest

import numpy as np

from shared.protocol import ProtocolError
from shared.tensor_serialization import decode_tensor, encode_tensor
from shared.video_protocol import SparseTokenDistribution, VideoProposalRequest


class VideoProtocolTests(unittest.TestCase):
    def test_tensor_round_trip(self):
        value = np.arange(12, dtype=np.float16).reshape(3, 4)
        np.testing.assert_array_equal(decode_tensor(encode_tensor(value)), value)

    def test_zlib_tensor_round_trip_reduces_repeated_frame_payload(self):
        value = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        encoded = encode_tensor(value, codec="zlib")
        self.assertLess(len(encoded.data), value.nbytes)
        np.testing.assert_array_equal(decode_tensor(encoded), value)

    def test_sparse_distribution_lengths_are_validated(self):
        request = VideoProposalRequest(
            task_id="v1", request_id="r1", sequence_no=0, base_revision=0,
            cache_position=0, route="cloud", committed_tokens=[],
            tokens=[SparseTokenDistribution(1, [1, 2], [0.8], 0.8)],
        )
        with self.assertRaises(ProtocolError):
            request.to_dict()


if __name__ == "__main__":
    unittest.main()
