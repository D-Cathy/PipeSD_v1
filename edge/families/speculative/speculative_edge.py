# families/speculative/speculative_edge.py
import time
import uuid

from core.roles import BaseInferenceRole
from families.speculative.trajectory import DraftTrajectory
from shared.protocol import FinalizeRequest, InitRequest, ProposalRequest, VerificationResponse
from shared.serialization import CONTENT_TYPE, pack_message


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
        init_bytes = pack_message(InitRequest(task_id=task_id, tokens=prefix_tokens))
        print(f"[Speculative] Initializing target verifier for task {task_id}")
        init_future = self.channel.submit(
            endpoint_url=init_url,
            data=init_bytes,
            headers={"Content-Type": CONTENT_TYPE},
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
        sequence_no = 0
        revision = 0
        current_batch_tokens = []
        current_batch_probs = []
        stop_reason = "max_tokens"

        while len(self.trajectory) < self.exp_cfg.max_generated_tokens:
            token, prob = self.draft_model.sample()
            self.trajectory.append_step(token, prob)
            current_batch_tokens.append(token)
            current_batch_probs.append(prob.tolist() if hasattr(prob, "tolist") else prob)
            self.collector.record_token_duration(0.04)

            draft_eos = hasattr(self.draft_model, "is_eos") and self.draft_model.is_eos(token)
            reached_limit = len(self.trajectory) >= self.exp_cfg.max_generated_tokens
            if not (self.strategy.check_verify_condition(len(current_batch_tokens)) or draft_eos or reached_limit):
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
            future = self.channel.submit(
                endpoint_url=f"{self.channel.config.server_url.rstrip('/')}/propose",
                data=pack_message(payload),
                headers={"Content-Type": CONTENT_TYPE},
                tag=task_tag,
            )

            verify_result = future.result()
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

            if hasattr(self.draft_model, "sync_generated_tokens"):
                self.draft_model.sync_generated_tokens(self.trajectory.tokens)
            else:
                self.draft_model.reset_kv_cache(len(self.trajectory))
            self.channel.drain_tag(task_tag)
            current_n_past = response.n_past
            revision = response.revision
            sequence_no += 1
            current_batch_tokens = []
            current_batch_probs = []
            if any(self.draft_model.is_eos(t) for t in self.trajectory.tokens[-2:]) if hasattr(self.draft_model, "is_eos") else False:
                stop_reason = "eos"
                break

        total_time = time.time() - start_time
        exit_payload = pack_message(FinalizeRequest(task_id=task_id))
        self.channel.submit(
            endpoint_url=f"{self.channel.config.server_url.rstrip('/')}/exit",
            data=exit_payload,
            headers={"Content-Type": CONTENT_TYPE},
        ).result()

        output_text = self.draft_model.decode(self.trajectory.tokens) if hasattr(self.draft_model, "decode") else str(self.trajectory.tokens)
        self.collector.record_custom("output_length", len(self.trajectory.tokens))
        self.collector.record_custom("stop_reason", stop_reason)
        self.collector.save_sample_result(task_id, output_text, 0, total_time, self.exp_cfg.algorithm)
        return self.trajectory.tokens
