import unittest

from pipesd.runtime import (
    Action,
    BackendNode,
    CollaborationContext,
    Decision,
    InProcessChannel,
    HTTPNode,
    NodeRequest,
    Strategy,
    Task,
)


class RuntimeContractTests(unittest.TestCase):
    def test_backend_node_adapts_existing_python_backend(self):
        class Backend:
            def add(self, left, right=0):
                return left + right

        node = BackendNode(Backend(), node_id="edge", location="local")
        result = node.execute(NodeRequest("add", args=(2,), kwargs={"right": 3}))

        self.assertEqual(result, 5)
        self.assertTrue(node.supports("add"))
        self.assertFalse(node.supports("missing"))
        self.assertEqual(node.health()["node_id"], "edge")

    def test_in_process_channel_dispatches_and_closes(self):
        channel = InProcessChannel({"/double": lambda value: value * 2})

        self.assertEqual(channel.request("/double", 4), 8)
        self.assertEqual(channel.submit("/double", 5).result(), 10)
        channel.close()
        self.assertEqual(channel.health()["status"], "closed")
        with self.assertRaises(RuntimeError):
            channel.request("/double", 4)

    def test_strategy_contract_uses_context_and_decision(self):
        class AlwaysCloud(Strategy):
            def decide(self, context):
                return Decision(Action.RUN_CLOUD, reason=context.task.modality)

        task = Task("t-1", "text", prompt="hello")
        decision = AlwaysCloud().decide(CollaborationContext(task))

        self.assertEqual(decision.action, Action.RUN_CLOUD)
        self.assertEqual(decision.reason, "text")

    def test_http_node_hides_remote_url_and_channel(self):
        channel = InProcessChannel({"http://cloud:8000/init": lambda payload: {"seen": payload}})
        node = HTTPNode("http://cloud:8000", channel, endpoints={"init": "/init"})

        self.assertEqual(node.invoke("init", b"hello"), {"seen": b"hello"})
        self.assertTrue(node.supports("init"))


if __name__ == "__main__":
    unittest.main()
