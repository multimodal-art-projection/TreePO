# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.lora.request import LoRARequest
import random

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

import torch.distributed as dist
class Rank0Filter(logging.Filter):
    """只允许 rank0 打印日志；在 DDP 未初始化时默认打印。"""
    def filter(self, record):
        return (not dist.is_available()                # PyTorch 没编译分布式
                or not dist.is_initialized()           # 还没 init_process_group
                or dist.get_rank() == 0)               # 已初始化且是 rank0
logger.addFilter(Rank0Filter())
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

# pre-requisite for tree inference
from recipe.treepo.vllm_rollout_tree import (
    _repeat_list_interleave,
    _repeat_list_by_indices, 
    _repeat_by_indices,
    _find_subtokens_idx,
    _find_all_subtokens_idx,
    _find_subtokens_idx_reverse,
    _get_generated_logprobs_from_vllm_output,
    _get_vllm_inputs,
    _has_repetition,
    SampleStatus,
    FinishedReason,
    FINISHED_REASON_PRIORITY,
    LOGPROG_PLACEHOLDER,
    DataSampleTree,
    get_vllm_inputs_from_samples,
    _increment_tree_idx_depth,
    compute_sequence_entropy_from_topk,
    extract_last_boxed,
    weight_to_discrete_allocate,
    _gumbel_topk_permutation,
    _gumbel_topk_permutation_list,
)
import copy
import time
from collections import defaultdict

class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        lora_kwargs = kwargs.pop('lora_kwargs', {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        
        tokenizer_vocab_size = len(list(tokenizer.get_vocab().values()))
        def fix_oov(token_ids, logits):
            logits[tokenizer_vocab_size:] = -float("inf")
            return logits
        kwargs["logits_processors"] = [fix_oov]

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if lora_int_ids:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=str(lora_int_id), lora_int_id=lora_int_id)] * batch_size
        start_time = time.time()
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)
            
            end_time = time.time()
            
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                # if "tools_kwargs" in non_tensor_batch.keys():
                #     non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                for key in non_tensor_batch.keys():
                    non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], self.sampling_params.n)
                    
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        total_time = end_time - start_time
        non_tensor_batch["dp_batch_total_time"] = np.array([total_time for _ in range(response_attention_mask.shape[0])], dtype=object)
        non_tensor_batch["dp_batch_n_trajectory"] = np.array([response_attention_mask.shape[0] for _ in range(response_attention_mask.shape[0])], dtype=object) # 保留这个数据，消除 dp 分配的影响
        non_tensor_batch["dp_batch_response_len"]  = np.array(response_attention_mask.sum(dim=1).cpu().numpy().tolist(), dtype=object)

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    def softmax(self, x, axis=None):
        x = np.asarray(x)
        x_max = np.max(x, axis=axis, keepdims=True)
        x_stable = x - x_max
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
    def create_fallback_sample_by_policy(
        self,
        new_sample, 
        max_token_per_step,
        fallback_policy,
        max_depth,
        cal_prob_block_size=128,
        fallback_window_size = None,
        max_fallback_window=-1,
    ):
        '''
        Fallback to certan index and truncate the response, than change the sample.status
        '''
        if fallback_window_size is None:
            fallback_window_size = max_token_per_step
        
        # total_infer_step_blocks = (new_sample.output_len + max_token_per_step - 1) // max_token_per_step  
        if fallback_policy == "random":
            # 随机 fallback 到某个位置
            # fallback_idx = np.random.randint(0, 1+max(0, (new_sample.output_len-1))//fallback_window_size) * fallback_window_size
            # Sample fallback_idx within the specified window if max_fallback_window > 0
            total_blocks = (new_sample.output_len + fallback_window_size - 1) // fallback_window_size
            if max_fallback_window > 0:
                fallback_block = np.random.randint(1, min(max_fallback_window, total_blocks)+1)
            else:
                # Sample from all possible blocks
                fallback_block = np.random.randint(1, total_blocks+1)
            
            fallback_idx = (total_blocks - fallback_block) * fallback_window_size # 特殊情况 0，短回复直接从头开始
            
        elif fallback_policy == "min_logprob_token":
            if max_fallback_window > 0:
                raise NotImplementedError()
            new_sample_response_logprobs = new_sample.logprobs
            new_sample_response_logprobs = [x[0] for x in new_sample_response_logprobs]
            fallback_idx = np.argmin(new_sample_response_logprobs)
        elif fallback_policy in ["block", "inverse_block"]:
            if max_fallback_window > 0:
                cal_prob_block_size = max_fallback_window
                logger.INFO(f"use {max_fallback_window=} to overwrite {cal_prob_block_size=}")
            new_sample_response_logprobs = new_sample.logprobs
            new_sample_response_logprobs = [x[0] for x in new_sample_response_logprobs]
            sub_block_logprobs_list = [new_sample_response_logprobs[i:i+cal_prob_block_size] for i in range(0, len(new_sample_response_logprobs), cal_prob_block_size)]
            if fallback_policy== "inverse_block":
                logprobs_mean_list = [np.mean(sub_block) for sub_block in sub_block_logprobs_list]
            elif fallback_policy== "block":
                logprobs_mean_list = [-np.mean(sub_block) for sub_block in sub_block_logprobs_list]
            logprobs_softmax_list = self.softmax(logprobs_mean_list)
            indices = list(range(len(logprobs_softmax_list)))
            selected_idx = random.choices(indices, weights=logprobs_softmax_list, k=1)[0]
            fallback_idx = 0
            for i in range(selected_idx):
                fallback_idx += len(sub_block_logprobs_list[i])
            # print(f"[fallback_idx: {fallback_idx}")  
        elif fallback_policy == "from_start":
            fallback_idx = 0
        else:
            raise NotImplementedError(f"fallback_policy {fallback_policy} not implemented")
        
        # total_infer_step_blocks
        n_remaining_infer_steps = int(np.ceil(fallback_idx/max_token_per_step))
        # 如果 fallback_idx 以后，深度还在最大上限，强行多切一些，使得 remaining steps = max_step - 1
        if n_remaining_infer_steps >= max_depth:
            fallback_idx = max_token_per_step * (max_depth-1)

        # 从 fallback_idx 开始，去掉后面的回复
        truncate_len = new_sample.output_len - fallback_idx
        new_sample.truncate_response(truncate_len=truncate_len)
        
        n_remaining_infer_steps = int(np.ceil(fallback_idx/max_token_per_step))
        new_sample.truncate_tree_idx(new_sample.depth-n_remaining_infer_steps)
        
        new_sample.status = SampleStatus.TO_INFER 
        return new_sample   
   
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences_tree_deepth_first_vanilla_mode(
        self, 
        prompts: DataProto, 
        fixed_step_width: int = 2,
        random_first_div_min: int = -1,
        random_first_div_max: int = -1,
        max_token_per_step: int = 512, 
        max_depth: int = 6,
        max_width: int = 8, # ~= rollout n trajectory
        fallback_window_size = None,
        fallback_traj_policy: str = 'finished_first',
        fallback_policy: str = "random",
        max_fallback_window:int = 3,
        divergence_policy: str = "fixed_avg",
        divergence_budget_control: str = "by_fixed_div", # or "by_infer_step_token_budget"
        logprob_div_temperature: float = 1.0,
        cal_prob_block_size: int = 128,
        min_width: int = -1,
        traj_filter_min_len: int = -1,
        max_response_token: int = -1, # limit of the sum of the response token per prompt
        total_budget_policy: str = "by_response_traj", # or "by_response_token"
        minimum_requests_per_gpu: int = -1, # TODO: 用于减少单机上的 inference bubble
        boxed_eos_policy:str = "boxed_as_eos", # "boxed_as_eos", "boxed_and_eos_first"
        repetition_es_policy:str = "least_prioritized", # "least_prioritized",  "most_prioritized"
        **kwargs) -> DataProto:
        
        """
        implement tree inference for model without SFT cold-start
        """
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        
        # assert total_budget_policy != "by_response_token", f"not supprt {total_budget_policy=} in this function, buggy"
        
        # assert max_depth * max_token_per_step <= self.max_output_len, f"only supports max_depth * max_token_per_step (current: {max_depth * max_token_per_step}) <= max_output_len (current: {self.max_output_len}) ATM"
        logger.debug(kwargs)
        calculate_entropy = kwargs.get("calculate_entropy", False)
        if calculate_entropy:
            assert kwargs.get("logprobs", -1) >= 0, "n_return_logprobs should be greater than 0 if calculate_entropy is true"
        
        if fallback_policy in ["min_logprob_token"]:
            assert calculate_entropy and kwargs.get("logprobs", -1) >= 0, "fallback_policy min_logprob_token requires calculate_entropy and logprobs > 0"
        
        random_first_div_min = kwargs.get("random_first_div_min", -1)
        random_first_div_max = kwargs.get("random_first_div_max", -1)
        logger.debug(f"set {random_first_div_max=}")
        
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        # pad_token_id = prompts.meta_info["pad_token_id"] # 这样传进来的  pad_token_id 是 None，不知道为什么，所以用 self.pad_token_id
        
        
        answer_start_tag_tokens = prompts.meta_info["answer_start_tag_tokens"]
        
        batch_size = idx.size(0)
        
        n_target_trajectory = len(prompts) * max_width
        non_tensor_batch = prompts.non_tensor_batch
        vllm_inputs = _get_vllm_inputs(prompts, self.pad_token_id)

        
        init_input_ids_by_root_node = {root_idx: vllm_input['prompt_token_ids'] for root_idx, vllm_input in enumerate(vllm_inputs)}
        # input_len_by_root_node = {root_idx: len(vllm_input['prompt_token_ids']) for root_idx, vllm_input in enumerate(vllm_inputs)}
        width_counter = {root_idx: 0 for root_idx in range(len(prompts))} # record number of finished samples for each root node
        finished_samples = {root_idx: [] for root_idx in range(len(prompts))} # store the tuples of (finished response ids, reasoning tree batch idx, finished_status)
        
        fallback_counter = defaultdict(int) # 记录 fallback 的次数
        n_finished_traj_at_first_fallback = defaultdict(int)
        
        # 构造第一批输入
        samples_to_infer = []
        for root_idx in range(len(prompts)):
            samples_to_infer.append(
                    DataSampleTree(
                        tree_idx=str(root_idx),
                        init_input_len=len(vllm_inputs[root_idx]['prompt_token_ids']),
                        input_ids=vllm_inputs[root_idx]['prompt_token_ids'],
                        status=SampleStatus.TO_INFER,
                        finished_reason=FinishedReason.UNFINISHED,
                )
            )
        # 第一次都做 rollout
        # 后面每次都要动态计算 traj 上限
        if total_budget_policy == "by_response_traj":
            # 按照预定的 trajectory 数量预定
            div_upper_bound = max_width
        elif total_budget_policy == "by_response_token":
            # 按照总的 response token 预算
            assert max_response_token >= max_token_per_step
            div_upper_bound = max_response_token//max_token_per_step
        else:
            raise NotImplementedError()

        if random_first_div_max  > div_upper_bound:
            logger.warning(f"[WARNING] {random_first_div_max=} is larger than {div_upper_bound=}, use {div_upper_bound=} instead.")
            random_first_div_max = div_upper_bound
                
        if divergence_budget_control == "by_infer_step_token_budget":
            logger.warning(f"[WARNING] setting {divergence_budget_control=}, now use the {fixed_step_width=} as the total maximum inference token budget, and {max_token_per_step=} as the budget for vLLM params for each input.")
            first_step_div = fixed_step_width//max_token_per_step//len(samples_to_infer)
            if first_step_div <= 1:
                logger.warning(f"[WARNING] average first divergence {first_step_div=} <=1, please set a larger value of {fixed_step_width=}.")
            first_step_div = max(1, first_step_div) # at least one
            first_step_div = min(first_step_div, div_upper_bound)
            samples_to_infer = _repeat_list_interleave(samples_to_infer, first_step_div)
        elif random_first_div_max > fixed_step_width:
            if random_first_div_min < 0:
                random_first_div_min = fixed_step_width
            assert random_first_div_max <= div_upper_bound, f"{random_first_div_max=} should be less than or equal to {div_upper_bound=}"
            duplicated_indices = []
            for sample in samples_to_infer:
                # 这里的 sample.tree_idx 是 str 类型
                first_step_div = np.random.randint(random_first_div_min, random_first_div_max+1) 
                duplicated_indices.extend([sample.root_node]*first_step_div)
            # 这里的 duplicated_indices 是 list[int]，每个元素是 sample.root_node 的索引
            samples_to_infer = _repeat_list_by_indices(samples_to_infer, duplicated_indices)
            logger.info(f"first step rollout, duplicate {len(vllm_inputs)} samples to infer: {len(samples_to_infer)}")
        elif "by_random_div" in divergence_budget_control:
            min_random_div, max_random_div = divergence_budget_control.split("_")[-1].split("to")
            min_random_div = int(min_random_div)
            max_random_div = int(max_random_div)
            assert min_random_div>=1 and min_random_div < max_random_div
            random_div_this_step = random.randint(min_random_div, max_random_div)
            samples_to_infer = _repeat_list_interleave(samples_to_infer, random_div_this_step)
        else:
            samples_to_infer = _repeat_list_interleave(samples_to_infer, fixed_step_width)
                
        next_infer_step = 1
        samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step=next_infer_step)
        
        start_time = time.time()
        while len(samples_to_infer) > 0:
            assert kwargs.get("n", 1) == 1, "n must be 1 for tree search, use repeat_interleave to generate multiple samples"
            kwargs["max_tokens"] = max_token_per_step
            
            # if isinstance(vllm_inputs[0], list):
            #     vllm_inputs = [{"prompt_token_ids": vllm_input} for vllm_input in vllm_inputs]
            vllm_inputs = get_vllm_inputs_from_samples(samples_to_infer)
            logger.debug(f"vllm_inputs to infer: {len(vllm_inputs)}")
            with self.update_sampling_params(**kwargs):
                # breakpoint()
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

            samples_last_step = copy.deepcopy(samples_to_infer)
            samples_to_infer = []
            
            # response_ids = [list(output.outputs[0].token_ids) for output in outputs]
            # response_strs = [self.tokenizer.decode(response_id, skip_special_tokens=False) for response_id in response_ids]

            samples_to_go_deeper = defaultdict(list)            
            finished_samples_this_step_by_root_node = defaultdict(int) # 记录当前 step 中每个 root node 的 finished 数量，用于做分岔预算转移
            
            for infer_batch_idx, (sample, output) in enumerate(zip(samples_last_step, outputs)):
                assert len(output.outputs) == 1, "vllm should only generate one output"
                # 取出当前 infer step 结果
                response_id = list(output.outputs[0].token_ids)
                response_str = output.outputs[0].text
                
                # if response_str.find("</answer>") >= 0:
                #     print(f"response_str: {response_str}")
                # debug：是否有多个 </answer>
                # if response_str.count("</answer>") > 1:
                #     print(f"response_str: {response_str}")
                #     print(f"reasoning tree batch idx: {sample.tree_idx}")
                #     print("--------------------------------")
                
                if calculate_entropy:
                    response_logprobs = _get_generated_logprobs_from_vllm_output(output.outputs[0])
                else:
                    response_logprobs = [[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))] * len(response_id)
                
                sample = copy.deepcopy(sample) # 必须要 copy，否则会影响后续的判断
                
                sample.extend_response(response_id, response_logprobs)
                    
                # answer_start_idx = _find_subtokens_idx(response_id, answer_start_tag_tokens)
                last_eos_idx = _find_subtokens_idx_reverse(response_id, [eos_token_id])
                
                boxed_answer = extract_last_boxed(response_str)

                actual_response = sample.full_response_token_ids
                acutal_response_len = sample.actual_response_len
                
                res_has_repretition, repeated_substr = _has_repetition(response_str)
                # # 上一轮强行要求作答的 query，不论有没有作答都要结束了
                # if sample.status == SampleStatus.FINISH_NEXT_INFER:
                #     # print(f"answer required to stop at last round, stop generation, response:\n{response_str}")
                #     # filter short and finished response
                #     if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                #         pass
                #     else:
                #         if boxed_answer is not None:
                #             sample.finished_reason = FinishedReason.FINISHED
                #             sample.status = SampleStatus.FINISHED
                #             finished_samples[sample.root_node].append(sample)
                #         else:
                #             sample.finished_reason = FinishedReason.UNCLOSED_ANSWER
                #             sample.status = SampleStatus.FINISHED
                #             finished_samples[sample.root_node].append(sample)
                #         width_counter[sample.root_node] += 1
                #         finished_samples_this_step_by_root_node[sample.root_node] += 1
                #     continue
                
                # 如果 response 中存在完整 \\boxed{}
                # el
                if repetition_es_policy == "most_prioritized" and res_has_repretition:
                    if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                        pass
                    else:
                        sample.finished_reason = FinishedReason.REPETITION_STOP
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                if (boxed_eos_policy == "boxed_as_eos" and boxed_answer is not None) or \
                    (boxed_eos_policy == "boxed_and_eos_first" and boxed_answer is not None and last_eos_idx >= 0) :
                    if np.random.random() < 0.0005:
                        logger.info(f"detect full answer, [response]:\n{response_str}")
                    
                    # filter short and finished response
                    if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                        pass
                    else:
                        sample.finished_reason = FinishedReason.FINISHED
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                elif last_eos_idx >= 0:
                    if np.random.random() < 0.001:
                        logger.info(f"detect EOS, but no answer, [response]:\n{response_str}")
                    # filter short and finished response
                    if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                        pass
                    else:
                        sample.finished_reason = FinishedReason.EARLY_STOPPED_BY_EOS
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                # 如果没有 answer，则认为这条路径没有结束，判断是否要继续 rollout
                # 如果实际 response 长度 >= max_output_len，则认为这条路径已经结束
                elif acutal_response_len >= self.config.response_length:
                    if np.random.random() < 0.001:
                        logger.debug(f"{sample.tree_idx}: {acutal_response_len=} >= max_output_len, stop,[response]:\n{response_str}")
                    if acutal_response_len > self.config.response_length:
                        logger.warning(f"{sample.tree_idx}: get {acutal_response_len=} > max_output_len={self.config.response_length}, force truncation\n[response before truncation]:\n{response_str}")
                        truncate_len = acutal_response_len - self.config.response_length
                        sample.truncate_response(truncate_len=truncate_len)
                    if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                        pass
                    else:
                        sample.finished_reason = FinishedReason.MAX_OUTPUT_TOKENS
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                elif sample.depth >= max_depth:
                    logger.info(f"{sample.tree_idx} has depth {sample.depth}: reach max_depth, stop")
                    if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                        pass
                    else:
                        sample.finished_reason = FinishedReason.MAX_INFER_STEPS
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                if res_has_repretition:
                    # print(f"{sample.tree_idx}: response contains repetition, stop, [response]:\n{repeated_substr[:100]} ...")
                    if traj_filter_min_len > 0 and sample.output_len < traj_filter_min_len:
                        pass
                    else:
                        sample.finished_reason = FinishedReason.REPETITION_STOP
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                else:
                    # if response_str.find("</answer>") >= 0 or self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                    #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                    # 如果还有足够的 token budget，则继续 rollout
                    assert sample.status == SampleStatus.TO_INFER, f"sample {sample.tree_idx} should be TO_INFER, but is {sample.status}"
                    samples_to_go_deeper[sample.root_node].append(copy.deepcopy(sample))
            
                
            root_to_infer_count = defaultdict(int) # 在加入到 samples_to_infer 时计数，方便用于后续 fallback，key 是 root_node: int
            if samples_to_go_deeper:
                logger.debug(f"[divergence] {len(samples_to_go_deeper)} roots to go deeper after step {next_infer_step}")
                for root_node in samples_to_go_deeper.keys():
                    logger.debug(f"[divergence] {root_node=} has {len(samples_to_go_deeper[root_node])} active paths after step {next_infer_step}")
                    if total_budget_policy == "by_response_traj": # 按照预定的 trajectory 数量预定
                        div_upper_bound = max_width
                    elif total_budget_policy == "by_response_token": # 按照总的 response token 预算
                        # 根据每个 root node 的实际 response token 数量，计算出一个上限
                        generated_tokens = 0
                        for finished_sample in finished_samples[root_node]:
                            generated_tokens += finished_sample.actual_response_len
                        # 根据 ongoing sample 的平均长度，和 infererence token，计算最多能有多少分岔。
                        avg_active_response_len = float(np.mean([todo_sample.actual_response_len for todo_sample in samples_to_go_deeper[root_node]]))
                        estitmated_reamining_div_budget = max(
                            int((max_response_token-generated_tokens)//(avg_active_response_len+max_token_per_step)),
                            0
                        )
                        logger.debug(f"[divergence] {root_node=}, {max_response_token=}, finished {width_counter[root_node]}, {generated_tokens=}, {avg_active_response_len=:.2f}, {max_token_per_step=}=> calcualte new {estitmated_reamining_div_budget=}")
                        
                        # 用预估的总 traj 数量，跟上下限比较：砍掉或者新增预算
                        estitmated_n_traj = estitmated_reamining_div_budget +  width_counter[root_node]
                        if estitmated_n_traj >= max_width:
                            div_upper_bound = max_width
                            logger.debug(f"[divergence] {root_node=} {estitmated_n_traj=} >= {max_width=} when consuming {generated_tokens} tokens, force set {div_upper_bound=} to generate less trajectories.")
                        # 如果算出来的 upper
                        elif estitmated_n_traj < min_width:
                            div_upper_bound = min_width
                            logger.debug(f"[divergence] {root_node=} {estitmated_n_traj=} < {min_width=} when consuming {generated_tokens} tokens, force set {div_upper_bound=} to generate more trajectories.")
                        else:
                            div_upper_bound = estitmated_n_traj
                            logger.debug(f"[divergence] {root_node=} use {estitmated_n_traj=} as div_upper_bound in step {next_infer_step=}.")
                            
                    # 计算 `total_divergence`，也就是本次 infer step 的分岔预算。
                    if divergence_budget_control == "by_infer_step_token_budget":
                        # the budget is pre-defined by total number of output tokens in `fixed_step_width`, the upper bound is definied by the final trajectory numbers.
                        token_budget_per_query = fixed_step_width//len(samples_to_go_deeper.keys()) # 这个是 cross-query 的？
                        remaining_width_budget = max(div_upper_bound - width_counter[root_node], 0)
                        total_divergence = min(
                            remaining_width_budget, 
                            token_budget_per_query//max_token_per_step
                        ) # 如果超过了 max_width，则不再继续 rollout
                    elif divergence_budget_control == "by_fixed_div":
                        # the budget is transferred from the finished trajectories, the upper bound is definied by the final trajectory numbers.
                        # 计算除了已完成+ active 路径 double 以外，还剩下多少预算；remaining_width_budget <= 0 是允许的，下面的 div policy 会让 active path 走完?
                        remaining_width_budget = max(div_upper_bound - width_counter[root_node], 0)
                        extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * fixed_step_width                    
                        total_divergence = min(
                            remaining_width_budget, 
                            extra_allow_divergence + fixed_step_width * len(samples_to_go_deeper[root_node])
                        )
                    # "by_random_div_mton"
                    elif "by_random_div" in divergence_budget_control:
                        random_div_this_step = random.randint(min_random_div, max_random_div) # including low and high end
                        remaining_width_budget = max(div_upper_bound - width_counter[root_node], 0)
                        extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * random_div_this_step                    
                        total_divergence = min(
                            remaining_width_budget, 
                            extra_allow_divergence + random_div_this_step * len(samples_to_go_deeper[root_node])
                        )
                    else:
                        raise NotImplementedError(f"{divergence_budget_control=} not implemented")
                    
                    total_divergence = max(total_divergence, len(samples_to_go_deeper[root_node])) # 每个路径至少要能继续往下分岔一个
                        
                    if divergence_policy == "fixed_avg":
                        logger.debug(f"[divergence] root {root_node}; already finished {width_counter[root_node]}; number of samples finishing in this step: {finished_samples_this_step_by_root_node[root_node]}; remaining width budget: {remaining_width_budget}; path to go deeper: {len(samples_to_go_deeper[root_node])}; divergence for next round {total_divergence=}")
                        average_divergence = total_divergence // len(samples_to_go_deeper[root_node])
                        remainder_divergence = total_divergence % len(samples_to_go_deeper[root_node])

                        # 将 total_divergence 的预算，均匀分配给 samples_to_go_deeper[root_node] 中的每个样本
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            if path_idx + 1 == len(samples_to_go_deeper[root_node]):
                                n_rollout = average_divergence + remainder_divergence
                            else:
                                n_rollout = average_divergence
                            n_rollout = max(1, n_rollout) # 至少给一个预算
                            for _ in range(n_rollout):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                # if self.tokenizer.decode(new_sample.input_ids[new_sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                                #     print(f"[ERROR] find additional</answer> in input_ids: {new_sample.input_ids}")
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    elif divergence_policy in ["logprob_weighted_div", "inverse_logprob_weighted_div"]:
                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                        cum_logprob = torch.tensor([lp.sum() for lp in log_probs])  # shape (N,)
                        # 先做一个 shift，防止极端负值下溢
                        if divergence_policy == "logprob_weighted_div":
                            shifted = -1.0 * (cum_logprob - cum_logprob.min())
                        elif divergence_policy == "inverse_logprob_weighted_div":
                            shifted = 1.0 * (cum_logprob - cum_logprob.min())  # 如果 logprob 越大，分岔权重越大
                        weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                        divergence_budgets = weight_to_discrete_allocate(weights, total_divergence)
                        for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                            for _ in range(div_budget):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    elif divergence_policy in ["norm_logprob_weighted_div",  "norm_inv_logprob_weighted_div"]:
                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                        cum_logprob = torch.tensor([lp.mean() for lp in log_probs])  # shape (N,)
                        # 先做一个 shift，防止极端负值下溢
                        if divergence_policy =="norm_logprob_weighted_div":
                            shifted = -1.0 * (cum_logprob - cum_logprob.min())
                        elif divergence_policy == "norm_inv_logprob_weighted_div":
                            shifted = 1.0 * (cum_logprob - cum_logprob.min())
                        weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                        divergence_budgets = weight_to_discrete_allocate(weights, total_divergence)
                        for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                            for _ in range(div_budget):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    elif divergence_policy in ["node_logprob_weighted_div", "inverse_node_logprob_weighted_div"]:
                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            traj_logprob = torch.tensor([x[0] for x in sample.logprobs])
                            node_logprob = traj_logprob[-min(max_token_per_step, traj_logprob.shape[0]):] # 只取最后一个  node 的概率，同时考虑 node 里面的长度不足的问题。
                            log_probs.append(node_logprob) 
                        cum_logprob = torch.tensor([lp.mean() for lp in log_probs])  # shape (N,)，这里要求 tokens in node 的平均值，避免被负值影响
                        # 先做一个 shift，防止极端负值下溢
                        if divergence_policy ==  "node_logprob_weighted_div":
                            shifted = -1.0 * (cum_logprob - cum_logprob.min())
                        elif divergence_policy ==  "inverse_node_logprob_weighted_div":
                            shifted = 1.0 * (cum_logprob - cum_logprob.min())
                        weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                        divergence_budgets = weight_to_discrete_allocate(weights, total_divergence)
                        for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                            for _ in range(div_budget):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    else:
                        raise NotImplementedError(f"divergence_policy {divergence_policy} not implemented")
            
            # 是否不做 fallback？
            for root_idx in width_counter: 
                if fallback_policy == "no_fallback":
                    # raise NotImplementedError("fallback_policy 'no_fallback' is not implemented yet, it will cause traj % dp !=0")
                    break
                # 在没有 active path 的情况下，才做 fallback
                # 逻辑是鼓励现有的长路径优先 infer 和 分岔，然后再回看先完成的。
                if root_to_infer_count[root_idx] > 0:
                    logger.debug(f"[fallback] {root_idx=} has active paths {root_to_infer_count[root_idx]=}, at {next_infer_step=}, skip fallback and continue generation.")
                    continue
                if width_counter[root_idx] + root_to_infer_count.get(root_idx, 0)>= max_width:
                    logger.debug(f"[fallback] {root_idx=} has finished {width_counter[root_idx]=} >= {max_width=} trajectories at {next_infer_step=}, skip fallback and wait.")
                    continue
                #  TODO：这里很复杂，后面要重写 fallback 逻辑
                if total_budget_policy == "by_response_traj": # 按照预定的 trajectory 数量预定
                    div_upper_bound = max_width
                    remaining_width_budget = div_upper_bound - (root_to_infer_count.get(root_idx, 0) + width_counter[root_idx])
                    
                elif total_budget_policy == "by_response_token": # 按照总的 response token 预算
                    # 根据每个 root node 的实际 response token 数量，计算出一个上限
                    finished_lens = []
                    for finished_sample in finished_samples[root_idx]:
                        finished_lens.append(finished_sample.actual_response_len)
                    avg_finished_len = np.mean(finished_lens)
                    # 预算不足一条，或还没有完成的 path
                    if max_response_token - sum(finished_lens) < avg_finished_len or len(finished_lens) <=0:
                        continue
                    # 根据 finished sample 的平均长度，和 infererence token，计算最多能有多少分岔。
                    estitmated_reamining_div_budget = int((max_response_token-sum(finished_lens))//avg_finished_len)
                    
                    estitmated_n_traj = estitmated_reamining_div_budget + width_counter[root_idx]
                    if estitmated_n_traj < min_width:
                        div_upper_bound = min_width
                        logger.info(f"[fallback] {root_idx=} {estitmated_n_traj=} < {min_width=} when consuming {generated_tokens} tokens, force set {div_upper_bound=} to generate more trajectories.")
                    elif estitmated_n_traj > max_width:
                        div_upper_bound = max_width
                        logger.info(f"[fallback] {root_idx=} has {estitmated_n_traj=} > {max_width=} trajectories when consuming {generated_tokens} tokens, set upperbound {div_upper_bound=}.")
                    else:
                        div_upper_bound = estitmated_n_traj
                        logger.info(f"[fallback] {root_idx=} has consumed {generated_tokens} tokens, use {estitmated_n_traj=} as div upperbound in fallback.")
                    remaining_width_budget = min(
                        div_upper_bound - (root_to_infer_count.get(root_idx, 0) + width_counter[root_idx]),
                        max_width - (root_to_infer_count.get(root_idx, 0) + width_counter[root_idx])
                    )
                
                # max_divergence_nodes = remaining_width_budget // fixed_step_width
                # 如果这个 query 没有深度搜索的 path，且还有宽度预算，则从 finished_samples 中取样
                # 如果剩余的 remaining_width_budget 很多，会不会抽样到非常类似的？是否应该做 no-fallback？
                if remaining_width_budget > 0:
                    # print(f"root_idx {root_idx} has remaining width budget, but no go-deeper path, build {remaining_width_budget} samples to infer")
                    # finished_samples[root_idx].sort(key=lambda x: int(x[2].split("_")[0]))
                    # 记录第一次发生 fallback 的时候有多少条正常跑完的 trajectory
                    if root_idx not in n_finished_traj_at_first_fallback:
                        n_finished_traj_at_first_fallback[root_idx] = root_to_infer_count.get(root_idx, 0) + width_counter[root_idx]
                    
                    FALLBACK_POLICY_SELECT = fallback_traj_policy
                    
                    selected_samples = copy.deepcopy(finished_samples[root_idx])
                    if FALLBACK_POLICY_SELECT == 'finished_first_strict':
                        selected_samples = sorted(selected_samples, key=lambda s: s.finished_reason)
                    elif FALLBACK_POLICY_SELECT == 'finished_first':
                        # 按完成权重排序，但引入随机性
                        selected_samples = _gumbel_topk_permutation(selected_samples)
                    elif FALLBACK_POLICY_SELECT == 'random_all':
                        selected_samples = copy.deepcopy(finished_samples[root_idx])
                        random.shuffle(selected_samples)
                    elif FALLBACK_POLICY_SELECT == 'soft_boxed_and_eos':
                        selected_samples = [sample for sample in selected_samples 
                                            if sample.finished_reason == FinishedReason.FINISHED or \
                                                        sample.finished_reason == FinishedReason.EARLY_STOPPED_BY_EOS
                                                ]
                        random.shuffle(selected_samples)
                    elif FALLBACK_POLICY_SELECT == 'longer_first':                                                
                        # finished_samples[root_idx] = _gumbel_topk_permutation(finished_samples[root_idx])
                        len_weights = [float(s.actual_response_len) for s in selected_samples]
                        # 默认权重越大越优先
                        selected_samples = _gumbel_topk_permutation_list(selected_samples, weights=len_weights, reverse=False)
                    # : 1.只选 complete  优先；2. 只选 min_len_blocks window 长度以上的 （256）；
                    elif "boxed_ge_window" in FALLBACK_POLICY_SELECT:
                        min_len_blocks = int(FALLBACK_POLICY_SELECT.split('_')[-1])
                        assert min_len_blocks > 0, f"{FALLBACK_POLICY_SELECT=} set error, legal example: boxed_ge_window_n"
                        selected_samples = [
                            sample for sample in selected_samples 
                            if sample.actual_response_len >= min_len_blocks*fallback_window_size and \
                                sample.finished_reason == FinishedReason.FINISHED
                        ]
                        random.shuffle(selected_samples)
                    else:
                        raise NotImplementedError(f"{FALLBACK_POLICY_SELECT=} not implemented")
                    
                    if len(selected_samples) <= 0:
                        logger.debug(f"{root_idx=} has no valid finished samples, create new input to rollout from start.")
                        dummy_sample = copy.deepcopy(finished_samples[root_idx][0])
                        selected_samples = [DataSampleTree(
                                tree_idx=str(dummy_sample.root_node),
                                init_input_len=dummy_sample.init_input_len,
                                input_ids=dummy_sample.init_input_ids,
                                status=SampleStatus.TO_INFER,
                                finished_reason=FinishedReason.UNFINISHED,
                        )]
                    for selected_sample in selected_samples:
                        new_sample = copy.deepcopy(selected_sample)
                        if ((fallback_window_size is not None and new_sample.output_len >= fallback_window_size) or \
                            (fallback_window_size is None)) and new_sample.output_len > 0:
                            new_sample = self.create_fallback_sample_by_policy(
                                new_sample, 
                                max_token_per_step=max_token_per_step, 
                                fallback_policy=fallback_policy, 
                                max_depth=max_depth,
                                cal_prob_block_size=cal_prob_block_size,
                                fallback_window_size=fallback_window_size,
                                max_fallback_window=max_fallback_window,
                            )
                        # 如果比一个 fallback window size 小，就从头 rollout
                        else:
                            logger.debug(f"[fallback] {root_idx=} has no valid finished samples longer than {fallback_window_size=}, create new input from start.")
                            new_sample = DataSampleTree(
                                    tree_idx=str(new_sample.root_node),
                                    init_input_len=new_sample.init_input_len,
                                    input_ids=new_sample.init_input_ids,
                                    status=SampleStatus.TO_INFER,
                                    finished_reason=FinishedReason.UNFINISHED,
                            )
                        
                        if divergence_budget_control == "by_fixed_div":
                            estimated_div_per_step_per_prompt = fixed_step_width
                        elif "by_random_div" in divergence_budget_control:
                            random_div_this_step = random.randint(min_random_div, max_random_div) # including low and high end
                            estimated_div_per_step_per_prompt = random_div_this_step
                        elif divergence_budget_control == "by_infer_step_token_budget":
                            estimated_div_per_step_per_prompt = fixed_step_width//(batch_size * max_token_per_step)
                        
                        if remaining_width_budget >= estimated_div_per_step_per_prompt:
                            for _ in range(estimated_div_per_step_per_prompt):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                                fallback_counter[root_idx] += 1
                            remaining_width_budget -= estimated_div_per_step_per_prompt
                        else:
                            for _ in range(remaining_width_budget):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                                fallback_counter[root_idx] += 1
                            remaining_width_budget = 0
                        # 如果 fallback_policy （回退到什么位置的参数）是 from_start
                        # 则模拟从头开始做 tree rollout，所以加完一次就直接结束
                        if fallback_policy == "from_start":
                            break
            
                # -------- 凑够最低 inference request，减少 bubble -----
                # Maximum concurrency for 4096 tokens per request: 181.91x
                if minimum_requests_per_gpu > 0:
                    raise NotImplementedError("low efficiency implementation, skip")
                    n_finished_paths = sum([len(finished_samples[root_idx]) for root_idx in finished_samples])
                    extra_div_to_fill  = minimum_requests_per_gpu - len(samples_to_infer)
                    
                    if n_finished_paths < batch_size:
                        logger.info(f"current {n_finished_paths=} < {batch_size=}, too less to fill {minimum_requests_per_gpu=}")
                    elif extra_div_to_fill > 0 and len(samples_to_infer) > 0:
                        logger.info(f"current number of infer request: {len(samples_to_infer)}, need to fill {extra_div_to_fill=}")
                        
                        # -------- 设定不触发补齐的条件，防止无限循环 --------
                        n_finished_query = 0
                        query_finish_status = [0.0 for root_idx in finished_samples] # 记录完成任务的比例，比如 rollout 16，实际有 20 条，则为 20/16=1.25
                        for root_idx in finished_samples:
                            if total_budget_policy == "by_response_traj":
                                if len(finished_samples[root_idx])>=max_width:
                                    n_finished_query+=1
                                query_finish_status[root_idx] = len(finished_samples[root_idx])/max_width
                            elif total_budget_policy == "by_response_token":
                                finished_tokens = sum([finished_sample.actual_response_len for finished_sample in finished_samples[root_idx]]) 
                                if finished_tokens>= max_response_token:
                                    n_finished_query+=1
                                query_finish_status[root_idx] = finished_tokens/max_response_token
                        
                        STOP_FILL_RATIO = 0.60
                        # if n_finished_query/batch_size >= 0.95:
                        if float(np.mean(query_finish_status[root_idx])) >= STOP_FILL_RATIO:
                            extra_div_to_fill = -1
                            minimum_requests_per_gpu=-1
                            logger.info(f"reaching {STOP_FILL_RATIO*100}% of the total inference budget, stop in-batch bubble filling")
                        
                        BUBLE_FILL_POLICY_SELECT = 'unfinished_first'
                        while extra_div_to_fill > 0:
                            
                            if BUBLE_FILL_POLICY_SELECT == "random":
                                # randomly pick paths to fallback
                                while True:
                                    root_idx = random.choice(list(finished_samples.keys()))
                                    # finished_samples[root_idx] = sorted(finished_samples[root_idx], key=lambda s: s.finished_reason)
                                    if finished_samples[root_idx]:
                                        selected_sample = random.choice(finished_samples[root_idx])
                                        break
                                    elif samples_to_go_deeper[root_idx]:
                                        selected_sample = random.choice(samples_to_go_deeper[root_idx])
                                        break
                            elif BUBLE_FILL_POLICY_SELECT == 'unfinished_first':
                                # 按权重排序，但引入随机性
                                sorted_root_idxs = _gumbel_topk_permutation_list(
                                    list(finished_samples.keys()),
                                    query_finish_status,
                                    reverse=True, # 完成度越低，优先级越高
                                    )
                                for root_idx in sorted_root_idxs:
                                    if finished_samples[root_idx]:
                                        selected_sample = random.choice(finished_samples[root_idx])
                                        break
                                    elif samples_to_go_deeper[root_idx]:
                                        selected_sample = random.choice(samples_to_go_deeper[root_idx])
                                        break
                            else:
                                raise NotImplementedError
                            
                            new_sample = copy.deepcopy(selected_sample)
                            new_sample = self.create_fallback_sample_by_policy(
                                new_sample, 
                                max_token_per_step=max_token_per_step, 
                                fallback_policy=fallback_policy, 
                                max_depth=max_depth,
                                cal_prob_block_size=cal_prob_block_size,
                                max_fallback_window=max_fallback_window,
                            )
                            estimated_div_per_step_per_prompt = min(fixed_step_width, extra_div_to_fill)
                            for _ in range(estimated_div_per_step_per_prompt):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                            extra_div_to_fill -= fixed_step_width

                # 对 request sample 做 sorting，hopefully 让相似的 request 更近
                samples_to_infer.sort(key= lambda sample: sample.root_node)
                
            next_infer_step = next_infer_step + 1
            # append the tree idx depth if there are still unfinished samples
            samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step)
        
        end_time = time.time()
        
        # decode the finished samples
        # for root_idx in finished_samples:
        #     print(f"========= root_idx: {root_idx}, n-trajectory: {len(finished_samples[root_idx])} =========")
        #     for finished_sample in finished_samples[root_idx]:
        #         print(self.tokenizer.decode(finished_sample.init_input_ids + finished_sample.full_response_token_ids, skip_special_tokens=False))
        #         print(f"reasoning tree batch idx: {finished_sample.tree_idx}")
        #         print(f"finished status: {finished_sample.status}, finished reason: {finished_sample.finished_reason}")
        #         print(f"cumulative logprobs: {sum(finished_sample.logprobs)}")
        #         print("--------------------------------")
        
        total_n_trajectory = sum(len(finished_samples[root_idx]) for root_idx in finished_samples)

        # 统计 finished_reason：
        finished_reason_counter = defaultdict(int)
        for root_idx in finished_samples:
            for finished_sample in finished_samples[root_idx]:
                if sample.output_len > self.config.response_length:
                    logger.warning(f"[response before truncation]:\n{self.tokenizer.decode(sample.full_response_token_ids, skip_special_tokens=False)}")
                    truncate_len = sample.output_len - self.config.response_length
                    sample.truncate_response(truncate_len=truncate_len)
                finished_reason_counter[finished_sample.finished_reason] += 1
        logger.info(f"For {total_n_trajectory} responses, [finished reason]:\n{finished_reason_counter}")
        
        # assert sum(len(finished_samples[root_idx]) for root_idx in finished_samples) == n_target_trajectory, "finished_samples should be equal to n_target_trajectory"

        # 对齐原有输出格式
        response = []
        tree_indices = []
        response_root_indices = []
        finish_infer_step = []
        count_again = defaultdict(int)
        for root_idx in range(len(prompts)):
            for sample in finished_samples[root_idx]:
                response.append(sample.full_response_token_ids)
                response_root_indices.append(root_idx) # 保存每一条 response 对应的 root idx，因为后面要用来做 repeat（数量不一致的情况不能直接 repeat interleave。）
                tree_indices.append(copy.copy(sample.tree_idx))
                finish_infer_step.append(float(sample.tree_idx.split('/')[-1].split('-')[0]))
                count_again[root_idx] += 1
        logger.info(f"minimum and maximum numbers of trajectories per query: {min(width_counter.values())}, {max(width_counter.values())}")
        logger.info(f"minimum and maximum numbers of trajectories per query (checked): {min(count_again.values())}, {max(count_again.values())}")
        assert len(count_again.values()) == len(prompts) #, f"Checking Missmatch:\n[count_again]\n{count_again}\n[width_counter]\n{width_counter}"
        logger.debug(f"Checking Missmatch:\n[count_again]\n{count_again}\n[width_counter]\n{width_counter}")
        
        print(f"time cost {end_time - start_time} seconds for {total_n_trajectory=}. average finish step: {float(np.mean(finish_infer_step)):.2f} average time per traj: {(end_time - start_time) / total_n_trajectory:.4f}s.")
        
        # breakpoint()
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        
        non_tensor_batch['n_finished_traj_at_first_fallback'] = np.array([n_finished_traj_at_first_fallback[root_idx] for root_idx in range(batch_size)], dtype=object)
        non_tensor_batch['fallback_counter'] = np.array([fallback_counter[root_idx] for root_idx in range(batch_size)], dtype=object)
        

        if max_width > 1 or max_response_token > max_token_per_step:            
            # use _repeat_by_indices to support different number of responses per root node
            idx = _repeat_by_indices(idx, response_root_indices)
            attention_mask = _repeat_by_indices(attention_mask, response_root_indices)
            position_ids = _repeat_by_indices(position_ids, response_root_indices)
            batch_size = idx.size(0)
            for key in non_tensor_batch.keys():
                non_tensor_batch[key] = _repeat_by_indices(non_tensor_batch[key], response_root_indices)  
        
        # return the tree_idx for further use in the trainer
        # already repeated
        non_tensor_batch["tree_idx"] = np.array(tree_indices, dtype=object)
        non_tensor_batch['traj_finish_step'] = np.array(finish_infer_step, dtype=object)

        
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if calculate_entropy:
            all_sequence_logprobs = [] # Store logprobs for later processing
            for root_idx in range(len(prompts)):
                for sample in finished_samples[root_idx]:
                    all_sequence_logprobs.append(sample.logprobs)
            assert len(all_sequence_logprobs) == batch_size, f"Logprob extraction produced {len(all_sequence_logprobs)} items, but current batch size is {batch_size}. Entropy will not be added."
            non_tensor_batch["response_logprobs"] = np.array(all_sequence_logprobs, dtype=object)

        non_tensor_batch["dp_batch_total_time"] = np.array([end_time - start_time for _ in range(response.shape[0])], dtype=object)
        non_tensor_batch["dp_batch_n_trajectory"] = np.array([total_n_trajectory for _ in range(response.shape[0])], dtype=object)
        non_tensor_batch["dp_batch_response_len"]  = np.array(response_attention_mask.sum(dim=1).cpu().numpy().tolist(), dtype=object)

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
