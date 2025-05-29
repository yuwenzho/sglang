# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with hpu graph."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import namedtuple
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import tqdm

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import is_hpu
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

_is_hpu = is_hpu()
if _is_hpu:

    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"

    from sglang.srt.hpu_utils import (
        SKIP_WARMUP,
        USE_CONTIGUOUS_PA,
        compute_hpu_attn_bias_decode,
        compute_hpu_attn_bias_prefill,
        get_decode_all_buckets,
        get_decode_batch_bucket,
        get_prefill_all_seq_len_buckets,
        get_prefill_seq_len_bucket,
        prepare_hpu_attn_bias_prefill,
        to_hpu_and_pad_1d,
        to_hpu_and_pad_1d_v2,
        create_hpu_block_metadata,
        HPUBlockMetadata
    )

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

HPUForwardBatch = namedtuple(
    "HPUForwardBatch",
    [
        "forward_mode",
        "batch_size",
        "input_ids",
        "out_cache_loc",
        "positions",
        "attn_bias",
        "seq_pos",
        "seq_idx",
        "valid_seq_len",
        "extend_seq_lens",
        "page_size",
        "block_list",
        "block_mapping",
        "block_groups",
        "block_usage",
        "attn_backend",
        "token_to_kv_pool",
        "use_contiguous_pa",
        "input_embeds",
        "extend_return_logprob",
        "padded_static_len",
        "capture_hidden_mode",
    ],
    defaults=[None, False, -1, CaptureHiddenMode.NULL],
)


def create_hpu_forward_batch(forward_batch: ForwardBatch, model_runner: ModelRunner):
    assert (
        forward_batch.hpu_metadata is not None
    ), "Expected HPU Metadata for HPU forward batch"
    batch_size = forward_batch.batch_size
    page_size = model_runner.token_to_kv_pool_allocator.page_size
    if forward_batch.forward_mode.is_extend():
        seq_len_list = forward_batch.extend_seq_lens
        sum_seq_len = seq_len_list.sum()
        max_prompt_len = get_prefill_seq_len_bucket(sum_seq_len)
        attn_bias, seq_pos, seq_idx = prepare_hpu_attn_bias_prefill(
            seq_lens=seq_len_list,
            max_prompt_len=max_prompt_len,
            dtype=model_runner.dtype,
        )
        attn_bias = attn_bias.to("hpu")
        seq_pos = seq_pos.to("hpu")
        seq_idx = seq_idx.to("hpu")
        padding_len = max_prompt_len - sum_seq_len
        max_prefill_seqs = model_runner.server_args.max_running_requests
        input_ids = to_hpu_and_pad_1d(forward_batch.input_ids, padding_len)
        positions = to_hpu_and_pad_1d(forward_batch.positions, padding_len)
        valid_seq_len = sum_seq_len.to("hpu", dtype=torch.int64)
        extend_seq_lens_padded = to_hpu_and_pad_1d(
            forward_batch.extend_seq_lens, max_prefill_seqs - batch_size
        )
        out_cache_loc = to_hpu_and_pad_1d(forward_batch.out_cache_loc, padding_len)
        batch_size = 1
        block_list = None
        block_mapping = None
        block_groups = None
        block_usage = None
        use_contiguous_pa = None
    else:
        padded_batch_size = get_decode_batch_bucket(batch_size)
        padding_len = padded_batch_size - batch_size
        input_ids = to_hpu_and_pad_1d(
            forward_batch.input_ids.to(torch.int64), padding_len
        )
        positions = to_hpu_and_pad_1d(
            forward_batch.positions.to(torch.int64), padding_len
        )
        valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int64, device="hpu")
        out_cache_loc = to_hpu_and_pad_1d(forward_batch.out_cache_loc, padding_len)
        batch_size = padded_batch_size
        attn_bias = compute_hpu_attn_bias_decode(
            page_size, forward_batch.hpu_metadata.block_usage, model_runner.dtype
        )

        seq_pos = None
        seq_idx = None
        extend_seq_lens_padded = None
        block_list = forward_batch.hpu_metadata.block_list.to("hpu")
        block_mapping = forward_batch.hpu_metadata.block_mapping.to("hpu")
        block_groups = forward_batch.hpu_metadata.block_groups.to("hpu")
        block_usage = forward_batch.hpu_metadata.block_usage.to("hpu")
        use_contiguous_pa = forward_batch.hpu_metadata.use_contiguous_pa

    return HPUForwardBatch(
        forward_mode=forward_batch.forward_mode,
        batch_size=batch_size,
        input_ids=input_ids,
        out_cache_loc=out_cache_loc,
        positions=positions,
        attn_bias=attn_bias,
        seq_pos=seq_pos,
        seq_idx=seq_idx,
        valid_seq_len=valid_seq_len,
        extend_seq_lens=extend_seq_lens_padded,
        page_size=page_size,
        block_list=block_list,
        block_mapping=block_mapping,
        block_groups=block_groups,
        block_usage=block_usage,
        attn_backend=forward_batch.attn_backend,
        token_to_kv_pool=forward_batch.token_to_kv_pool,
        use_contiguous_pa=use_contiguous_pa,
    )


def create_hpu_forward_batch_v2(forward_batch: ForwardBatch, model_runner: ModelRunner):
    assert (
        forward_batch.hpu_metadata is not None
    ), "Expected HPU Metadata for HPU forward batch"
    batch_size = forward_batch.batch_size
    page_size = model_runner.token_to_kv_pool_allocator.page_size
    if forward_batch.forward_mode.is_extend():
        seq_len_list = forward_batch.extend_seq_lens
        sum_seq_len = seq_len_list.sum()
        max_prompt_len = get_prefill_seq_len_bucket(sum_seq_len)
        attn_bias, seq_pos, seq_idx = prepare_hpu_attn_bias_prefill(
            seq_lens=seq_len_list,
            max_prompt_len=max_prompt_len,
            dtype=model_runner.dtype,
        )
        max_prefill_seqs = model_runner.server_args.max_running_requests
        input_ids = to_hpu_and_pad_1d_v2(forward_batch.input_ids, max_prompt_len, sum_seq_len)
        positions = to_hpu_and_pad_1d_v2(forward_batch.positions, max_prompt_len, sum_seq_len)
        valid_seq_len = sum_seq_len.to(torch.int64)
        extend_seq_lens_padded = to_hpu_and_pad_1d_v2(
            forward_batch.extend_seq_lens, max_prefill_seqs, batch_size
        )
        out_cache_loc = to_hpu_and_pad_1d_v2(forward_batch.out_cache_loc, max_prompt_len, sum_seq_len)
        block_list = None
        block_mapping = None
        block_groups = None
        block_usage = None
        use_contiguous_pa = None
        
        return HPUForwardBatch(
            forward_mode=forward_batch.forward_mode,
            batch_size=batch_size,
            input_ids=input_ids.to("hpu", non_blocking=True),
            out_cache_loc=out_cache_loc.to("hpu", non_blocking=True),
            positions=positions.to("hpu", non_blocking=True),
            attn_bias=attn_bias.to("hpu", non_blocking=True),
            seq_pos=seq_pos.to("hpu", non_blocking=True),
            seq_idx=seq_idx.to("hpu", non_blocking=True),
            valid_seq_len=valid_seq_len.to("hpu", non_blocking=True),
            extend_seq_lens=extend_seq_lens_padded.to("hpu", non_blocking=True),
            page_size=page_size,
            block_list=block_list,
            block_mapping=block_mapping,
            block_groups=block_groups,
            block_usage=block_usage,
            attn_backend=forward_batch.attn_backend,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            use_contiguous_pa=use_contiguous_pa,
        )
    else:
        padded_batch_size = get_decode_batch_bucket(batch_size)
        input_ids = to_hpu_and_pad_1d_v2(
            forward_batch.input_ids.to(torch.int64), padded_batch_size, batch_size
        )
        positions = to_hpu_and_pad_1d_v2(
            forward_batch.positions.to(torch.int64), padded_batch_size, batch_size
        )
        valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int64)
        out_cache_loc = to_hpu_and_pad_1d_v2(forward_batch.out_cache_loc, padded_batch_size, batch_size)
        batch_size = padded_batch_size
        attn_bias = compute_hpu_attn_bias_decode(
            page_size, forward_batch.hpu_metadata.block_usage, model_runner.dtype
        )
        seq_pos = None
        seq_idx = None
        extend_seq_lens_padded = None
        block_list = forward_batch.hpu_metadata.block_list
        block_mapping = forward_batch.hpu_metadata.block_mapping
        block_groups = forward_batch.hpu_metadata.block_groups
        block_usage = forward_batch.hpu_metadata.block_usage
        use_contiguous_pa = forward_batch.hpu_metadata.use_contiguous_pa

        return HPUForwardBatch(
            forward_mode=forward_batch.forward_mode,
            batch_size=batch_size,
            input_ids=input_ids.to("hpu", non_blocking=True),
            out_cache_loc=out_cache_loc.to("hpu", non_blocking=True),
            positions=positions.to("hpu", non_blocking=True),
            attn_bias=attn_bias.to("hpu", non_blocking=True),
            seq_pos=seq_pos,
            seq_idx=seq_idx,
            valid_seq_len=valid_seq_len.to("hpu", non_blocking=True),
            extend_seq_lens=extend_seq_lens_padded,
            page_size=page_size,
            block_list=block_list.to("hpu", non_blocking=True),
            block_mapping=block_mapping.to("hpu", non_blocking=True),
            block_groups=block_groups.to("hpu", non_blocking=True),
            block_usage=block_usage.to("hpu", non_blocking=True),
            attn_backend=forward_batch.attn_backend,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            use_contiguous_pa=use_contiguous_pa,
        )

def create_hpu_dummy_batch_prefill(
    seq_len, dtype, page_size, max_running_requests, attn_backend, token_to_kv_pool
):
    return HPUForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(seq_len, dtype=torch.int64, device="hpu"),
        out_cache_loc=torch.arange(seq_len, dtype=torch.int64, device="hpu"),
        positions=torch.zeros(seq_len, dtype=torch.int64, device="hpu"),
        attn_bias=torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device="hpu"),
        seq_pos=torch.zeros(1, seq_len, dtype=torch.int64, device="hpu"),
        seq_idx=torch.zeros(1, seq_len, dtype=torch.int64, device="hpu"),
        valid_seq_len=torch.ones((), dtype=torch.int64, device="hpu"),
        extend_seq_lens=torch.ones(
            max_running_requests,
            dtype=torch.int32,
            device="hpu",
        ),
        page_size=page_size,
        block_list=None,
        block_mapping=None,
        block_groups=None,
        block_usage=None,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        use_contiguous_pa=None,
    )

def create_hpu_dummy_batch_prefill_v2(
    seq_len, vocab_size
):
    forward_batch = ModelWorkerBatch(
        bid=1, 
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.zeros(seq_len, dtype=torch.int64, device="cpu"),
        req_pool_indices=torch.zeros(seq_len, dtype=torch.int64, device="cpu"), # 
        seq_lens=torch.tensor([seq_len], dtype=torch.int64, device="cpu"),
        seq_lens_cpu=None,
        out_cache_loc=torch.arange(seq_len, dtype=torch.int64, device="cpu"),
        seq_lens_sum=seq_len,
        sampling_info=SamplingBatchInfo(
            temperatures=torch.ones(seq_len, 1, dtype=torch.float32, device="cpu"),
            top_ps=torch.ones(seq_len, dtype=torch.float32, device="cpu"),
            top_ks=torch.ones(seq_len, dtype=torch.int32, device="cpu"),
            min_ps=torch.zeros(seq_len, dtype=torch.float32, device="cpu"),
            is_all_greedy=True,
            need_min_p_sampling=False,
            vocab_size=vocab_size
            ),
        return_logprob=False,
        top_logprobs_nums= None,
        token_ids_logprobs= None,
        global_num_tokens= None,
        global_num_tokens_for_logprob= None,
        can_run_dp_cuda_graph= False,
        extend_num_tokens= seq_len,
        extend_seq_lens= [seq_len],
        extend_prefix_lens=[0],
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids= None,
        multimodal_inputs= [None],
        encoder_cached= None,
        encoder_lens= None,
        encoder_lens_cpu= None,
        encoder_out_cache_loc= None,
        lora_paths= [None],
    )
    forward_batch.hpu_metadata = HPUBlockMetadata()
    return forward_batch


def create_hpu_dummy_batch_decode(
    batch_size, block_num, dtype, page_size, attn_backend, token_to_kv_pool
):
    return HPUForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        out_cache_loc=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        positions=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        attn_bias=torch.zeros(block_num, page_size, dtype=dtype, device="hpu"),
        seq_pos=None,
        seq_idx=None,
        valid_seq_len=torch.ones(batch_size, dtype=torch.int64, device="hpu"),
        extend_seq_lens=None,
        page_size=page_size,
        block_list=torch.zeros(block_num, dtype=torch.int64, device="hpu"),
        block_mapping=torch.zeros(block_num, batch_size, dtype=dtype, device="hpu"),
        block_groups=torch.zeros(block_num, dtype=torch.int64, device="hpu"),
        block_usage=torch.zeros(block_num, dtype=dtype, device="hpu"),
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        use_contiguous_pa=USE_CONTIGUOUS_PA,
    )

def create_hpu_dummy_batch_decode_v2(
    batch_size, block_num, dtype,vocab_size
):
    forward_batch = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.DECODE,
        input_ids=torch.zeros(batch_size, dtype=torch.int64, device="cpu"),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64, device="cpu"), # 
        seq_lens=torch.zeros(batch_size, dtype=torch.int64, device="cpu"),
        seq_lens_cpu=None,
        out_cache_loc=torch.zeros(batch_size, dtype=torch.int64, device="cpu"),
        seq_lens_sum=1,
        sampling_info=SamplingBatchInfo(
            temperatures=torch.ones(batch_size, 1, dtype=torch.float32, device="cpu"),
            top_ps=torch.ones(batch_size, dtype=torch.float32, device="cpu"),
            top_ks=torch.ones(batch_size, dtype=torch.int32, device="cpu"),
            min_ps=torch.zeros(batch_size, dtype=torch.float32, device="cpu"),
            is_all_greedy=True,
            need_min_p_sampling=False,
            vocab_size=vocab_size
            ),
        return_logprob=False,
        top_logprobs_nums= None,
        token_ids_logprobs= None,
        global_num_tokens= None,
        global_num_tokens_for_logprob= None,
        can_run_dp_cuda_graph= False,
        extend_num_tokens=None,
        extend_seq_lens=None,
        extend_prefix_lens= None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids= None,
        multimodal_inputs= [None],
        encoder_cached= None,
        encoder_lens= None,
        encoder_lens_cpu= None,
        encoder_out_cache_loc= None,
        lora_paths= [None],
    )
    forward_batch.hpu_metadata = HPUBlockMetadata(
        use_contiguous_pa=USE_CONTIGUOUS_PA,
        block_list=torch.zeros(block_num, dtype=torch.int64, device="cpu"),
        block_mapping=torch.zeros(block_num, batch_size, dtype=dtype, device="cpu"),
        block_groups=torch.zeros(block_num, dtype=torch.int64, device="cpu"),
        block_usage=torch.zeros(block_num, dtype=dtype, device="cpu"),
    )
    return forward_batch
    


class HPUAdapter:

    def __init__(self, model, dtype) -> None:
        self.model = model
        self.dtype = dtype

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        assert len(args) == 3, "Only three arguments are supported"
        input_batch = args[2]
        if input_batch.forward_mode.is_extend():
            input_batch.attn_bias.copy_(
                compute_hpu_attn_bias_prefill(
                    input_batch.seq_pos, input_batch.seq_idx, self.dtype
                )
            )
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HPUGraphRunner:
    """A HPUGraphRunner runs the forward pass of a model with HPU graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        import habana_frameworks.torch as htorch
        import vllm_hpu_extension.environment as environment
        from vllm_hpu_extension.flags import enabled_flags

        environment.runtime_params["model_type"] = (
            model_runner.model_config.hf_config.model_type
        )

        self.is_lazy = 1 if htorch.utils.internal.is_lazy() else 0
        if self.is_lazy:
            self.model = htorch.hpu.wrap_in_hpu_graph(
                HPUAdapter(self.model_runner.model, self.model_runner.dtype),
                disable_tensor_cache=True,
            )
        elif self.model_runner.server_args.enable_torch_compile:
            self.regional_compilation_layers_list = [RMSNorm, VocabParallelEmbedding]
            self.model = HPUAdapter(self.model_runner.model, self.model_runner.dtype)
            self._regional_compilation(self.model)
        else:
            self.model = HPUAdapter(self.model_runner.model, self.model_runner.dtype)
            logger.info("Running on Eager mode.")

        # Capture
        self.seen_configs: set = set()
        if not SKIP_WARMUP:
            try:
                with self.model_capture_mode(), torch._dynamo.utils.disable_cache_limit():
                    logger.info(
                        "Begin to capture hpu graph, you can use `export SGLANG_HPU_SKIP_WARMUP=true` to skip this step."
                    )
                    time_start = time.perf_counter()
                    self.capture()
                    time_end = time.perf_counter()
                    logger.info(
                        f"Capture hpu graph time: {time_end - time_start} seconds"
                    )
                    logger.info("Capture hpu graph success")
            except RuntimeError as e:
                raise Exception(f"Capture hpu graph failed: {e}\n")

    def _regional_compilation(self, module, parent_module=None, module_name=None):
        if isinstance(module, torch.nn.ModuleList):
            for children_name, children_module in module.named_children():
                self._compile_region(module, children_name, children_module)
        elif any(
            isinstance(module, layer) for layer in self.regional_compilation_layers_list
        ):
            self._compile_region(parent_module, module_name, module)
        else:
            for children_name, children_module in module.named_children():
                self._regional_compilation(children_module, module, children_name)

    def _compile_region(self, model, name, module):
        module = torch.compile(module, backend="hpu_backend", dynamic=False)
        setattr(model, name, module)

    @contextmanager
    def model_capture_mode(self):
        yield

    def can_run(self, forward_batch: ForwardBatch):
        return True

    def capture(self):
        # prefill
        time_start = time.perf_counter()
        # prefill_seq_len_buckets = get_prefill_all_seq_len_buckets()
        prefill_seq_len_buckets = [1024]
        # prefill_seq_len_buckets = [3072, 5120]
        # prefill_seq_len_buckets = [1024, 2048, 3072, 4096, 5120, 6144] # server 2048/128
        # prefill_seq_len_buckets = [1024, 2048, 3072, 4096, 5120]# native
        for seq_len in prefill_seq_len_buckets:
            self.capture_prefill(seq_len)
            self.seen_configs.add(("prefill", seq_len))
        time_end = time.perf_counter()
        logger.info(f"Capture prefill time: {time_end - time_start} seconds")

        # decode
        if self.model_runner.is_generation:
            time_start = time.perf_counter()
            # all_buckets = get_decode_all_buckets()
            all_buckets = [(4, 128)]
            # all_buckets = [(2, 128), (4, 128)]
            # all_buckets = [(1,128), (64, 1024), (128, 2176), (128, 2304)] # server 2048/128
            # all_buckets = [(1,128), (1, 384), (1, 512), (2, 512),(128, 384), (128, 512), 
            #                (96, 512), (64, 512), (32, 512),
            #                (16, 512), (8, 512), (8, 384), (4, 512), (4, 384), (2, 384), (1, 256)] # native
            for batch_size, seq_len in all_buckets:
                self.capture_decode(batch_size, seq_len)
                self.seen_configs.add(("decode", batch_size, seq_len))
            time_end = time.perf_counter()
            logger.info(f"Capture decode time: {time_end - time_start} seconds")

    def capture_prefill(self, seq_len):
        logger.info(f"Capture prefill with seq_len: {seq_len}")
        
        model_worker_batch = create_hpu_dummy_batch_prefill_v2(
            seq_len, self.model_runner.model_config.vocab_size)
        for i in range(3):
            self.forward_batch_generation(model_worker_batch)

    def capture_decode(self, batch_size, block_num):
        logger.info(
            f"Capture decode with batch_size: {batch_size} and block_num: {block_num}"
        )
        
        model_worker_batch = create_hpu_dummy_batch_decode_v2(
            batch_size, 
            block_num,
            self.model_runner.dtype,
            self.model_runner.model_config.vocab_size
        )
        for i in range(3):
            self.forward_batch_generation(model_worker_batch)


    def _forward(self, forward_batch: ForwardBatch):
        import habana_frameworks.torch as htorch

        forward_batch_hpu = create_hpu_forward_batch_v2(forward_batch, self.model_runner)
        self._check_config(forward_batch_hpu)
        results = self.model.forward(
            forward_batch_hpu.input_ids, forward_batch_hpu.positions, forward_batch_hpu
        )
        htorch.core.mark_step()
        if isinstance(results, LogitsProcessorOutput):
            output = LogitsProcessorOutput(
            next_token_logits=results.next_token_logits,
            hidden_states=(
                results.hidden_states
                if results.hidden_states is not None
                else None
                ),
            )
        elif isinstance(results, EmbeddingPoolerOutput):
            output = EmbeddingPoolerOutput(
                embeddings=results.embeddings.clone()[: forward_batch.batch_size]
            )
        return output

    def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        return self._forward(forward_batch)

    def _check_config(self, forward_batch):
        cfg: Optional[tuple] = None
        if forward_batch.forward_mode.is_extend():
            cfg = ("prefill", len(forward_batch.input_ids))
        else:
            cfg = ("decode", len(forward_batch.input_ids), len(forward_batch.block_list))
        print("config", cfg)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen:
            logger.warning("Configuration: %s was not warmed-up!", cfg)
    
    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        skip_sample: bool = False,
    ) -> Tuple[LogitsProcessorOutput, Optional[torch.Tensor]]:
        import habana_frameworks.torch as htorch
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        htorch.core.mark_step()
        logits_output = self.replay(forward_batch)

        if model_worker_batch.launch_done is not None:
            model_worker_batch.launch_done.set()

        htorch.core.mark_step()
        if skip_sample:
            next_token_ids = None
        else:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
            htorch.core.mark_step()

        return logits_output, next_token_ids

            
@contextmanager
def track_graph_compile(name: str):
    import habana_frameworks.torch as htorch
    from habana_frameworks.torch.hpu.metrics import metric_localcontext
    with metric_localcontext("graph_compilation") as gc:
        yield
        htorch.hpu.synchronize()
    if gc.stats()[0][1] != 0:
        msg = f"[{name}] graph compilation detected: {gc.stats()}"
        logger.warning(msg)