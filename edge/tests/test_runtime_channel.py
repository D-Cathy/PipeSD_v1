import unittest
from unittest.mock import Mock

from edge.core.channel import NetworkChannel
from edge.core.config import ChannelConfig
from pipesd.runtime.channel import Channel


class RuntimeChannelCompatibilityTests(unittest.TestCase):
    def test_network_channel_implements_public_channel_contract(self):
        channel = NetworkChannel(ChannelConfig(server_url="http://cloud.invalid"))
        try:
            self.assertIsInstance(channel, Channel)
            channel.submit = Mock(return_value=Mock(result=Mock(return_value={"ok": True})))
            self.assertEqual(channel.request("/test", b"payload"), {"ok": True})
        finally:
            channel.close()


if __name__ == "__main__":
    unittest.main()
