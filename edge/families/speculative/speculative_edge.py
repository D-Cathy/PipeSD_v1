# families/speculative/speculative_edge.py
import json
import time

import msgpack

from core.roles import BaseInferenceRole
from families.speculative.trajectory import DraftTrajectory


class SpeculativeEdgeRole(BaseInferenceRole):
    def __init__(self, model_node, channel, strategy, collector, model_config, exp_cfg):
        super().__init__(model_config=model_config)
        self.draft_model = model_node
        self.channel = channel
        self.strategy = strategy
        self.collector = collector
        self.exp_cfg = exp_cfg
        self.trajectory = DraftTrajectory()

    def load_model(self):
        print("[Speculative] Loading draft model backend...")
        if hasattr(self.draft_model, "load_model"):
            self.draft_model.load_model()
        print("[Speculative] Draft model backend is ready.")

    def _init_target(self, task_id, prompt):
        if hasattr(self.draft_model, "start_task"):
            prefix_tokens = self.draft_model.start_task(prompt)
        else:
            prefix_tokens = [1]

        init_url = f"{self.channel.config.server_url.rstrip('/')}/init"
        init_payload = {"type": "init", "task_id": task_id, "tokens": prefix_tokens}
        init_bytes = json.dumps(init_payload).encode("utf-8")
        print(f"[Speculative] Initializing target verifier for task {task_id}")
        init_future = self.channel.submit(
            endpoint_url=init_url,
            data=init_bytes,
            headers={"Content-Type": "application/json"},
        )
        init_res = init_future.result()
        if not isinstance(init_res, dict) or "error" in init_res:
            raise RuntimeError(f"Target verifier init failed: {init_res}")
        return int(init_res.get("n_past", 0))

    def process_task(self, task_id, prompt):
        self.trajectory.clear()
        self.draft_model.reset_kv_cache()
        start_time = time.time()
        current_n_past = self._init_target(task_id, prompt)

        task_tag = str(task_id)
        current_batch_tokens = []
        current_batch_probs = []

        while len(self.trajectory) < self.exp_cfg.max_generated_tokens:
            token, prob = self.draft_model.sample()
            self.trajectory.append_step(token, prob)
            current_batch_tokens.append(token)
            current_batch_probs.append(prob.tolist() if hasattr(prob, "tolist") else prob)
            self.collector.record_token_duration(0.04)

            if not self.strategy.check_verify_condition(self.trajectory):
                continue

            verify_base_len = len(self.trajectory) - len(current_batch_tokens)
            sent_tokens = current_batch_tokens.copy()
            sent_probs = current_batch_probs.copy()
            payload = {
                "type": "propose",
                "task_id": task_id,
                "tokens": sent_tokens,
                "probs": sent_probs,
                "n_past": current_n_past,
                "index": verify_base_len,
                "should_verify": True,
            }
            future = self.channel.submit(
                endpoint_url=f"{self.channel.config.server_url.rstrip('/')}/propose",
                data=msgpack.packb(payload),
                headers={"Content-Type": "application/msgpack"},
                tag=task_tag,
            )

            while not future.done() and len(self.trajectory) < self.exp_cfg.max_generated_tokens:
                wait_token, wait_prob = self.draft_model.sample()
                self.trajectory.append_step(wait_token, wait_prob)
                current_batch_tokens.append(wait_token)
                current_batch_probs.append(wait_prob.tolist() if hasattr(wait_prob, "tolist") else wait_prob)
                time.sleep(0.04)

            verify_result = future.result()
            if "error" in verify_result:
                raise RuntimeError(f"Target verifier failed: {verify_result}")

            accept_len = int(verify_result.get("n_accepted", verify_result.get("accept_length", 0)))
            final_token = verify_result.get("final_token")
            self.collector.record_verification(len(sent_tokens), accept_len)

            keep_len = verify_base_len + accept_len
            self.trajectory.rollback(keep_len)
            if final_token is not None and len(self.trajectory) < self.exp_cfg.max_generated_tokens:
                self.trajectory.append_step(final_token, None)

            reset_len = len(self.trajectory)
            self.draft_model.reset_kv_cache(reset_len)
            self.channel.drain_tag(task_tag)
            current_n_past = int(verify_result.get("n_past", current_n_past + accept_len + 1))
            current_batch_tokens = []
            current_batch_probs = []

        total_time = time.time() - start_time
        exit_payload = msgpack.packb({"type": "exit", "task_id": task_id})
        self.channel.submit(
            endpoint_url=f"{self.channel.config.server_url.rstrip('/')}/exit",
            data=exit_payload,
            headers={"Content-Type": "application/msgpack"},
        ).result()

        self.collector.save_sample_result(task_id, str(self.trajectory.tokens), 0, total_time, self.exp_cfg.algorithm)
        return self.trajectory.tokens
