# families/speculative/speculative_edge.py
import time
import uuid

from edge.core.roles import BaseInferenceRole
from edge.families.speculative.trajectory import DraftTrajectory
from shared.protocol import FinalizeRequest, InitRequest, ProposalRequest, VerificationResponse
from shared.serialization import CONTENT_TYPE, pack_message
from pipesd.runtime.node import HTTPNode, ensure_node
from pipesd.runtime import Action, CollaborationContext, Engine, Result, Task


class SpeculativeEdgeRole(BaseInferenceRole, Engine):
    def __init__(self, model_node, channel, strategy, collector, model_config, exp_cfg):
        super().__init__(model_config=model_config)
        self.edge_node = ensure_node(model_node, node_id="text-draft", location="edge")
        # Kept for downstream compatibility; orchestration calls go through edge_node.
        self.draft_model = getattr(self.edge_node, "backend", model_node)
        self.channel = channel
        self.cloud_node = HTTPNode(
            channel.config.server_url,
            channel,
            node_id="text-target",
            endpoints={"init": "/init", "propose": "/propose", "exit": "/exit"},
        )
        self.strategy = strategy
        self.collector = collector
        self.exp_cfg = exp_cfg
        self.trajectory = DraftTrajectory()
        self._loaded = False
        self.last_result = None

    def load_model(self):
        print("[Speculative] Loading draft model backend...")
        if self.edge_node.supports("load_model"):
            self.edge_node.invoke("load_model")
        self._loaded = True
        print("[Speculative] Draft model backend is ready.")

    def run(self, task):
        if not isinstance(task, Task):
            raise TypeError("SpeculativeEdgeRole.run expects a pipesd.Task.")
        if task.modality != "text":
            raise ValueError(f"Text speculative engine cannot run modality {task.modality!r}.")
        if not self._loaded:
            self.load_model()
        prompt = task.prompt or str(task.input_data or "")
        self.process_task(task.task_id, prompt)
        return self.last_result

    def _init_target(self, task_id, prompt):
        if self.edge_node.supports("start_task"):
            prefix_tokens = self.edge_node.invoke("start_task", prompt)
        else:
            prefix_tokens = [1]

        init_bytes = pack_message(InitRequest(task_id=task_id, tokens=prefix_tokens))
        print(f"[Speculative] Initializing target verifier for task {task_id}")
        init_res = self.cloud_node.invoke(
            "init", init_bytes, headers={"Content-Type": CONTENT_TYPE},
        )
        if not isinstance(init_res, dict) or "error" in init_res:
            raise RuntimeError(f"Target verifier init failed: {init_res}")
        return int(init_res.get("n_past", 0))

    def process_task(self, task_id, prompt):
        self.trajectory.clear()
        self.edge_node.invoke("reset_kv_cache")
        start_time = time.time()
        current_n_past = self._init_target(task_id, prompt)

        task_tag = str(task_id)
        sequence_no = 0
        revision = 0
        current_batch_tokens = []
        current_batch_probs = []
        stop_reason = "max_tokens"

        while len(self.trajectory) < self.exp_cfg.max_generated_tokens:
            token, prob = self.edge_node.invoke("sample")
            self.trajectory.append_step(token, prob)
            current_batch_tokens.append(token)
            current_batch_probs.append(prob.tolist() if hasattr(prob, "tolist") else prob)
            self.collector.record_token_duration(0.04)

            draft_eos = self.edge_node.supports("is_eos") and self.edge_node.invoke("is_eos", token)
            reached_limit = len(self.trajectory) >= self.exp_cfg.max_generated_tokens
            if hasattr(self.strategy, "decide"):
                decision = self.strategy.decide(CollaborationContext(
                    Task(task_id, "text", prompt=prompt),
                    state={
                        "pending_length": len(current_batch_tokens),
                        "draft_eos": draft_eos,
                        "reached_limit": reached_limit,
                    },
                ))
                should_verify = decision.action == Action.SEND_TO_CLOUD
            else:
                # Compatibility for third-party strategies written against the original API.
                should_verify = self.strategy.check_verify_condition(len(current_batch_tokens))
                should_verify = should_verify or draft_eos or reached_limit
            if not should_verify:
                continue

            verify_base_len = len(self.trajectory) - len(current_batch_tokens)
            sent_tokens = current_batch_tokens.copy()
            sent_probs = current_batch_probs.copy()
            request_id = uuid.uuid4().hex
            payload = ProposalRequest(
                task_id=task_id,
                request_id=request_id,
                sequence_no=sequence_no,
                base_revision=revision,
                tokens=sent_tokens,
                probs=sent_probs,
                n_past=current_n_past,
                index=verify_base_len,
            )
            verify_result = self.cloud_node.invoke(
                "propose", pack_message(payload),
                headers={"Content-Type": CONTENT_TYPE}, tag=task_tag,
            )
            response = VerificationResponse.from_dict(verify_result)
            if response.task_id != task_id or response.request_id != request_id:
                raise RuntimeError("Cloud response does not match the active request.")

            accept_len = response.n_accepted
            final_token = response.final_token
            self.collector.record_verification(len(sent_tokens), accept_len)

            keep_len = verify_base_len + accept_len
            self.trajectory.rollback(keep_len)
            if final_token is not None and len(self.trajectory) < self.exp_cfg.max_generated_tokens:
                self.trajectory.append_step(final_token, None)

            if self.edge_node.supports("sync_generated_tokens"):
                self.edge_node.invoke("sync_generated_tokens", self.trajectory.tokens)
            else:
                self.edge_node.invoke("reset_kv_cache", len(self.trajectory))
            self.channel.drain_tag(task_tag)
            current_n_past = response.n_past
            revision = response.revision
            sequence_no += 1
            current_batch_tokens = []
            current_batch_probs = []
            if any(self.edge_node.invoke("is_eos", t) for t in self.trajectory.tokens[-2:]) if self.edge_node.supports("is_eos") else False:
                stop_reason = "eos"
                break

        total_time = time.time() - start_time
        exit_payload = pack_message(FinalizeRequest(task_id=task_id))
        self.cloud_node.invoke(
            "exit", exit_payload, headers={"Content-Type": CONTENT_TYPE},
        )

        output_text = self.edge_node.invoke("decode", self.trajectory.tokens) if self.edge_node.supports("decode") else str(self.trajectory.tokens)
        self.collector.record_custom("output_length", len(self.trajectory.tokens))
        self.collector.record_custom("stop_reason", stop_reason)
        self.collector.save_sample_result(task_id, output_text, 0, total_time, self.exp_cfg.algorithm)
        self.last_result = Result(
            task_id=task_id,
            output=output_text,
            stop_reason=stop_reason,
            metrics=dict(getattr(self.collector, "current_metrics", {})),
            metadata={"tokens": list(self.trajectory.tokens), "modality": "text"},
        )
        return self.trajectory.tokens


class TextSpeculativeEngine(SpeculativeEdgeRole):
    """Public name for the text orchestration engine."""
