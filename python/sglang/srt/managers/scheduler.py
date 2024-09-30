"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A scheduler that manages a tensor parallel GPU worker."""

import json
import logging
import multiprocessing
import os
import time
import warnings
from typing import List, Optional, Union

import torch
import zmq

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    TokenizedRewardReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    BaseFinishReason,
    ImageInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
    configure_logger,
    is_generation_model,
    is_multimodal_model,
    kill_parent_process,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

# Crash on warning if we are running CI tests
crash_on_warning = os.getenv("SGLANG_IS_IN_CI", "false") == "true"


class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch

        # Init inter-process communication
        context = zmq.Context(2)

        if self.tp_rank == 0:
            self.recv_from_tokenizer = context.socket(zmq.PULL)
            self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.scheduler_port}")

            self.send_to_detokenizer = context.socket(zmq.PUSH)
            self.send_to_detokenizer.connect(
                f"tcp://127.0.0.1:{port_args.detokenizer_port}"
            )
        else:
            self.recv_from_tokenizer = self.send_to_detokenizer = None

        # Init tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=json.loads(server_args.json_model_override_args),
        )

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.model_config.hf_config.architectures):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
        self.is_generation = is_generation_model(
            self.model_config.hf_config.architectures, self.server_args.is_embedding
        )

        # Launch a tensor parallel worker
        self.tp_worker = TpModelWorker(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            server_args=server_args,
            nccl_port=port_args.nccl_ports[0],
        )
        self.tp_cpu_group = self.tp_worker.model_runner.tp_group.cpu_group

        # Get token and memory info from the tp worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_input_len,
            self.random_seed,
        ) = self.tp_worker.get_token_and_memory_info()
        set_random_seed(self.random_seed)

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init cache
        self.req_to_token_pool = self.tp_worker.model_runner.req_to_token_pool
        self.token_to_kv_pool = self.tp_worker.model_runner.token_to_kv_pool

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                disable=server_args.disable_radix_cache,
            )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache)

        # Init running status
        self.waiting_queue: List[Req] = []
        self.running_batch: ScheduleBatch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        self.current_inflight_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the FSM cache for constrained generation
        if not server_args.skip_tokenizer_init:
            self.regex_fsm_cache = FSMCache(
                server_args.tokenizer_path,
                {
                    "tokenizer_mode": server_args.tokenizer_mode,
                    "trust_remote_code": server_args.trust_remote_code,
                },
                skip_tokenizer_init=server_args.skip_tokenizer_init,
                constrained_json_whitespace_pattern=server_args.constrained_json_whitespace_pattern,
            )
        self.jump_forward_cache = JumpForwardCache()

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.min_new_token_ratio = min(
            global_config.base_min_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.new_token_ratio = self.min_new_token_ratio
        self.new_token_ratio_decay = global_config.new_token_ratio_decay
        self.do_not_get_new_batch = False

    def event_loop(self):
        while True:
            # Receive requests
            if self.tp_rank == 0:
                recv_reqs = self.recv_requests_from_zmq()
            else:
                recv_reqs = None

            # Process requests
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
            self.process_requests(recv_reqs)

            # Forward
            self.forward_step()

            # Send results
            if self.tp_rank == 0:
                for obj in self.out_pyobjs:
                    self.send_to_detokenizer.send_pyobj(obj)
                self.out_pyobjs = []

    def recv_requests_from_zmq(self):
        recv_reqs = []

        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        return recv_reqs

    def process_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                self.handle_generate_request(recv_req)
                self.do_not_get_new_batch = False
            elif isinstance(
                recv_req, (TokenizedEmbeddingReqInput, TokenizedRewardReqInput)
            ):
                self.handle_embedding_request(recv_req)
                self.do_not_get_new_batch = False
            elif isinstance(recv_req, FlushCacheReq):
                self.flush_cache()
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req)
            elif isinstance(recv_req, UpdateWeightReqInput):
                success, message = self.update_weights(recv_req)
                self.out_pyobjs.append(UpdateWeightReqOutput(success, message))
            else:
                raise ValueError(f"Invalid request: {recv_req}")

    @torch.inference_mode()
    def forward_step(self):
        if self.do_not_get_new_batch and self.current_inflight_req is None:
            new_batch = None
        else:
            new_batch = self.get_new_prefill_batch()
        self.do_not_get_new_batch = False

        if new_batch is not None:
            # Run a new prefill batch
            self.forward_prefill_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run a decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(global_config.num_continue_decode_steps):
                    self.num_generated_tokens += len(self.running_batch.reqs)
                    self.forward_decode_batch(self.running_batch)

                    # Print stats
                    if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
                        self.print_decode_stats()

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

                    if self.out_pyobjs and self.running_batch.has_stream:
                        break
            else:
                self.check_memory()
                self.new_token_ratio = global_config.init_new_token_ratio

    def print_decode_stats(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        throughput = self.num_generated_tokens / (time.time() - self.last_stats_tic)
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        logger.info(
            f"Decode batch. "
            f"#running-req: {len(self.running_batch.reqs)}, "
            f"#token: {num_used}, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

    def check_memory(self):
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if available_size != self.max_total_num_tokens:
            warnings.warn(
                "Warning: "
                f"available_size={available_size}, max_total_num_tokens={self.max_total_num_tokens}\n"
                "KV cache pool leak detected!"
            )
            exit(1) if crash_on_warning else None

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            warnings.warn(
                "Warning: "
                f"available req slots={len(self.req_to_token_pool.free_slots)}, "
                f"total slots={self.req_to_token_pool.size}\n"
                "Memory pool leak detected!"
            )
            exit(1) if crash_on_warning else None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            lora_path=recv_req.lora_path,
        )
        req.tokenizer = self.tokenizer

        # Image inputs
        if recv_req.image_inputs is not None:
            req.image_inputs = ImageInputs.from_dict(
                recv_req.image_inputs, self.model_config.vocab_size
            )
            req.origin_input_ids = self.tp_worker.model_runner.model.pad_input_ids(
                req.origin_input_ids_unpadded, req.image_inputs
            )

        req.return_logprob = recv_req.return_logprob
        req.top_logprobs_num = recv_req.top_logprobs_num
        req.stream = recv_req.stream
        req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len == -1:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(recv_req.input_ids) - 1

        # Init regex FSM
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
        ):
            if req.sampling_params.json_schema is not None:
                req.regex_fsm, computed_regex_string = self.regex_fsm_cache.query(
                    ("json", req.sampling_params.json_schema)
                )
            elif req.sampling_params.regex is not None:
                req.regex_fsm, computed_regex_string = self.regex_fsm_cache.query(
                    ("regex", req.sampling_params.regex)
                )
            if not self.disable_regex_jump_forward:
                req.jump_forward_map = self.jump_forward_cache.query(
                    computed_regex_string
                )

        # Truncate prompts that are too long
        if len(req.origin_input_ids) >= self.max_req_input_len:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_input_len - 1 - len(req.origin_input_ids),
        )

        self.waiting_queue.append(req)

    def handle_embedding_request(
        self,
        recv_req: Union[TokenizedEmbeddingReqInput, TokenizedRewardReqInput],
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
        )
        req.tokenizer = self.tokenizer

        # Truncate prompts that are too long
        if len(req.origin_input_ids) >= self.max_req_input_len:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

        self.waiting_queue.append(req)

    def get_new_prefill_batch(self) -> Optional[ScheduleBatch]:
        running_bs = (
            len(self.running_batch.reqs) if self.running_batch is not None else 0
        )
        if running_bs >= self.max_running_requests:
            return None

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        num_mixed_running = running_bs if self.is_mixed_chunk else 0

        adder = PrefillAdder(
            self.tree_cache,
            self.running_batch,
            self.new_token_ratio,
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size(),
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            num_mixed_running,
        )

        has_inflight = self.current_inflight_req is not None
        if self.current_inflight_req is not None:
            self.current_inflight_req.init_next_round_input(
                None if prefix_computed else self.tree_cache
            )
            self.current_inflight_req = adder.add_inflight_req(
                self.current_inflight_req
            )

        if self.lora_paths is not None:
            lora_set = (
                set([req.lora_path for req in self.running_batch.reqs])
                if self.running_batch is not None
                else set([])
            )

        for req in self.waiting_queue:
            if (
                self.lora_paths is not None
                and len(
                    lora_set
                    | set([req.lora_path for req in adder.can_run_list])
                    | set([req.lora_path])
                )
                > self.max_loras_per_batch
            ):
                break

            if adder.no_remaining_tokens():
                break
            req.init_next_round_input(None if prefix_computed else self.tree_cache)
            res = adder.add_one_req(req)
            if (
                not res
                or running_bs + len(adder.can_run_list) >= self.max_running_requests
            ):
                break

        can_run_list = adder.can_run_list

        if adder.new_inflight_req is not None:
            assert self.current_inflight_req is None
            self.current_inflight_req = adder.new_inflight_req

        if len(can_run_list) == 0:
            return None

        # Print stats
        if self.tp_rank == 0:
            if isinstance(self.tree_cache, RadixCache):
                self.tree_cache_metrics["total"] += (
                    adder.log_input_tokens + adder.log_hit_tokens
                ) / 10**9
                self.tree_cache_metrics["hit"] += (adder.log_hit_tokens) / 10**9
                tree_cache_hit_rate = (
                    self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
                )
            else:
                tree_cache_hit_rate = 0.0

            num_used = self.max_total_num_tokens - (
                self.token_to_kv_pool.available_size()
                + self.tree_cache.evictable_size()
            )

            if num_mixed_running > 0:
                logger.info(
                    f"Prefill batch"
                    f"(mixed #running-req: {num_mixed_running}). "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
                    f"#queue-req: {len(self.waiting_queue) - len(can_run_list) + has_inflight}"
                )
            else:
                logger.info(
                    f"Prefill batch. "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
                    f"#running-req: {running_bs}, "
                    f"#queue-req: {len(self.waiting_queue) - len(can_run_list) + has_inflight}"
                )

        # Return the new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
        )
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_list]
        return new_batch

    def forward_prefill_batch(self, batch: ScheduleBatch):
        # Build batch tensors
        batch.prepare_for_extend(self.model_config.vocab_size)

        decoding_reqs = []
        if self.is_mixed_chunk and self.running_batch is not None:
            self.running_batch.prepare_for_decode()
            batch.mix_with_running(self.running_batch)
            decoding_reqs = self.running_batch.reqs
            self.running_batch = None

        if self.is_generation:
            # Forward and sample the next tokens
            if batch.extend_num_tokens != 0:
                forward_batch = batch.get_forward_batch()
                logits_output, next_token_ids = self.tp_worker.forward_batch_generation(
                    forward_batch, batch
                )
                batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    next_token_ids
                )

                # Move logprobs to cpu
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs[
                            torch.arange(
                                len(next_token_ids), device=next_token_ids.device
                            ),
                            next_token_ids,
                        ].tolist()
                    )
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.tolist()
                    )
                    logits_output.normalized_prompt_logprobs = (
                        logits_output.normalized_prompt_logprobs.tolist()
                    )

                next_token_ids = next_token_ids.tolist()
            else:
                if self.tokenizer is None:
                    next_token_ids = []
                    for req in batch.reqs:
                        next_token_ids.append(
                            next(iter(req.sampling_params.stop_token_ids))
                        )
                else:
                    next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)

            # Check finish conditions
            logprob_pt = 0
            for i, req in enumerate(batch.reqs):
                if req is not self.current_inflight_req:
                    # Inflight reqs' prefill is not finished
                    req.completion_tokens_wo_jump_forward += 1
                    req.output_ids.append(next_token_ids[i])
                    req.check_finished()

                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.get_next_state(
                        req.regex_fsm_state, next_token_ids[i]
                    )

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                elif req not in decoding_reqs:
                    # To reduce overhead, only cache prefill reqs
                    self.tree_cache.cache_unfinished_req(req)

                if req is self.current_inflight_req:
                    # Inflight request would get a new req idx
                    self.req_to_token_pool.free(req.req_pool_idx)

                if req.return_logprob:
                    logprob_pt += self.add_logprob_return_values(
                        i, req, logprob_pt, next_token_ids, logits_output
                    )
        else:
            assert batch.extend_num_tokens != 0
            forward_batch = batch.get_forward_batch()
            embeddings = self.tp_worker.forward_batch_embedding(forward_batch)

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                req.embedding = embeddings[i]
                if req is not self.current_inflight_req:
                    # Inflight reqs' prefill is not finished
                    # dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                else:
                    self.tree_cache.cache_unfinished_req(req)

                if req is self.current_inflight_req:
                    # Inflight request would get a new req idx
                    self.req_to_token_pool.free(req.req_pool_idx)

        self.handle_finished_requests(batch)

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
        num_input_logprobs = req.extend_input_len - req.extend_logprob_start_len

        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.input_token_logprobs is None:
            input_token_logprobs = output.input_token_logprobs[
                pt : pt + num_input_logprobs - 1 - req.last_update_decode_tokens
            ]
            input_token_ids = req.fill_ids[
                len(req.fill_ids)
                - num_input_logprobs
                + 1 : len(req.fill_ids)
                - req.last_update_decode_tokens
            ]
            req.input_token_logprobs = list(zip(input_token_logprobs, input_token_ids))

            if (
                req.logprob_start_len == 0
            ):  # The first token does not have logprob, pad it.
                req.input_token_logprobs = [
                    (None, req.fill_ids[0])
                ] + req.input_token_logprobs

        if req.last_update_decode_tokens != 0:
            # Some decode tokens are re-computed in an extend batch
            req.output_token_logprobs.extend(
                list(
                    zip(
                        output.input_token_logprobs[
                            pt
                            + num_input_logprobs
                            - 1
                            - req.last_update_decode_tokens : pt
                            + num_input_logprobs
                            - 1
                        ],
                        req.fill_ids[
                            len(req.fill_ids)
                            - req.last_update_decode_tokens : len(req.fill_ids)
                        ],
                    )
                )
            )

        if req.top_logprobs_num > 0:
            if req.input_top_logprobs is None:
                req.input_top_logprobs = output.input_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.input_top_logprobs = [None] + req.input_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.output_top_logprobs.extend(
                    output.input_top_logprobs[i][-req.last_update_decode_tokens :]
                )
            req.output_top_logprobs.append(output.output_top_logprobs[i])

        return num_input_logprobs

    def forward_decode_batch(self, batch: ScheduleBatch):
        # Check if decode out of memory
        if not batch.check_decode_mem():
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode()
            self.new_token_ratio = new_token_ratio

            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.waiting_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        # Check for jump-forward
        if not self.disable_regex_jump_forward:
            jump_forward_reqs = batch.check_for_jump_forward(
                self.tp_worker.model_runner
            )
            self.waiting_queue.extend(jump_forward_reqs)
            if batch.is_empty():
                return

        # Update batch tensors
        self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
        batch.prepare_for_decode()

        # Forward and sample the next tokens
        forward_batch = batch.get_forward_batch()
        logits_output, next_token_ids = self.tp_worker.forward_batch_generation(
            forward_batch, batch
        )
        batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            next_token_ids
        )

        # Move logprobs to cpu
        if logits_output.next_token_logprobs is not None:
            next_token_logprobs = logits_output.next_token_logprobs[
                torch.arange(len(next_token_ids), device=next_token_ids.device),
                next_token_ids,
            ].tolist()

        next_token_ids = next_token_ids.tolist()

        # Check finish condition
        has_finished = False
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req.completion_tokens_wo_jump_forward += 1
            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.regex_fsm is not None:
                req.regex_fsm_state = req.regex_fsm.get_next_state(
                    req.regex_fsm_state, next_token_id
                )

            if req.finished():
                self.tree_cache.cache_finished_req(req)
                has_finished = True

            if req.return_logprob:
                req.output_token_logprobs.append(
                    (next_token_logprobs[i], next_token_id)
                )
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs.append(logits_output.output_top_logprobs[i])

        if not has_finished:
            self.do_not_get_new_batch = True

        self.handle_finished_requests(batch)

    def handle_finished_requests(self, batch: ScheduleBatch):
        output_rids = []
        output_meta_info = []
        output_finished_reason: List[BaseFinishReason] = []
        if self.is_generation:
            output_vids = []
            decoded_texts = []
            output_read_ids = []
            output_read_offsets = []
            output_skip_special_tokens = []
            output_spaces_between_special_tokens = []
        else:  # for embedding model
            output_embeddings = []
        unfinished_indices = []

        for i, req in enumerate(batch.reqs):
            if not req.finished() and req is not self.current_inflight_req:
                unfinished_indices.append(i)

            if req.finished() or (
                req.stream
                and (
                    self.decode_forward_ct % self.stream_interval == 0
                    or len(req.output_ids) == 1
                )
            ):
                output_rids.append(req.rid)
                output_finished_reason.append(req.finished_reason)
                if self.is_generation:
                    output_vids.append(req.vid)
                    decoded_texts.append(req.decoded_text)
                    read_ids, read_offset = req.init_incremental_detokenize()
                    output_read_ids.append(read_ids)
                    output_read_offsets.append(read_offset)
                    output_skip_special_tokens.append(
                        req.sampling_params.skip_special_tokens
                    )
                    output_spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )

                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                        "completion_tokens": len(req.output_ids),
                        "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                        "finish_reason": (
                            req.finished_reason.to_json()
                            if req.finished_reason is not None
                            else None
                        ),
                    }
                    if req.return_logprob:
                        (
                            meta_info["input_token_logprobs"],
                            meta_info["output_token_logprobs"],
                            meta_info["input_top_logprobs"],
                            meta_info["output_top_logprobs"],
                            meta_info["normalized_prompt_logprob"],
                        ) = (
                            req.input_token_logprobs,
                            req.output_token_logprobs,
                            req.input_top_logprobs,
                            req.output_top_logprobs,
                            req.normalized_prompt_logprob,
                        )
                    output_meta_info.append(meta_info)
                else:  # for embedding model
                    output_embeddings.append(req.embedding)
                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                    }
                    output_meta_info.append(meta_info)

        # Send to detokenizer
        if output_rids:
            if self.is_generation:
                self.out_pyobjs.append(
                    BatchTokenIDOut(
                        output_rids,
                        output_vids,
                        decoded_texts,
                        output_read_ids,
                        output_read_offsets,
                        output_skip_special_tokens,
                        output_spaces_between_special_tokens,
                        output_meta_info,
                        output_finished_reason,
                    )
                )
            else:  # for embedding model
                self.out_pyobjs.append(
                    BatchEmbeddingOut(
                        output_rids,
                        output_embeddings,
                        output_meta_info,
                        output_finished_reason,
                    )
                )

        # Remove finished reqs: update batch tensors
        batch.filter_batch(unfinished_indices)

    def flush_cache(self):
        if len(self.waiting_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.regex_fsm_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.waiting_queue[to_del]

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid:
                    req.finished_reason = FINISH_ABORT()
                    break

    def update_weights(self, recv_req: UpdateWeightReqInput):
        success, message = self.tp_worker.update_weights(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return success, message


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pipe_writer: multiprocessing.connection.Connection,
):
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    suppress_other_loggers()

    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank)
        pipe_writer.send("ready")
        scheduler.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()