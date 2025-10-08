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
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import numpy as np
import math
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import (get_response_mask,
                                         pad_2d_list_to_length)
from verl.workers.rollout.base import BaseRollout



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


import argparse
import os
import time

import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto

import copy
from collections import defaultdict

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


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


# from verl.workers.rollout.vllm_rollout import (vllm_mode, vLLMAsyncRollout, vLLMRollout)
# from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
#     _pre_process_inputs, _repeat_interleave)

def _repeat_by_indices(value: Union[torch.Tensor, np.ndarray], indices: List[int]) -> Union[torch.Tensor, List[Any]]:
    '''
    Repeat the value by the indices. E.g., _repeat_by_indices(value=[0, 1, 2, 3], indices=[0, 1, 2, 2, 3]) -> [0, 1, 2, 2, 3]
    Select (and thus 'repeat') elements of `value` along the first axis
    according to `indices`.

    Parameters
    ----------
    value : torch.Tensor | np.ndarray
        1-D or N-D data whose first dimension will be indexed.
    indices : List[int]
        Indices to pick, in the desired order.  Can contain duplicates.

    Returns
    -------
    torch.Tensor | List[Any]
        If `value` is a Tensor, a Tensor is returned.
        Otherwise a Python list is returned (so it behaves like the example).
    '''
    if isinstance(value, torch.Tensor):
        # Ensure indices live on the same device and are long dtype
        idx = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        # index_select gathers rows (axis 0) in the given order
        return value.index_select(dim=0, index=idx)

    # NumPy (or list/sequence converted to NumPy)
    arr = np.asarray(value)
    # Fancy-indexing does exactly what we need
    return arr[indices]

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _repeat_np_interleave(tensor, repeat_times):
    return np.repeat(tensor, repeat_times, axis=0)

def _repeat_list_interleave(any_list, repeat_times):
    # return [item for sublist in [[item] * repeat_times for item in any_list] for item in sublist]
    return [copy.deepcopy(item) for sublist in [[item] * repeat_times for item in any_list] for item in sublist]

def _repeat_list_by_indices(any_list, indices: List[int]) -> List[Any]:
    """
    Repeat the elements of `any_list` according to the `indices`.
    E.g., _repeat_list_by_indices(any_list=[0, 1, 2, 3], indices=[0, 1, 2, 2, 3]) -> [0, 1, 2, 2, 3]
    要求使用 deepcopy
    """
    idx_set = set(indices)
    # check all the indices are included and follow ascending order
    if not all(i in idx_set for i in range(len(any_list))):
        raise ValueError("All indices must be within the range of the list length.")
    if not all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1)):
        raise ValueError("Indices must be in ascending order.")
    return [copy.deepcopy(any_list[i]) for i in indices]

def _append_tree_idx_depth(reasoning_tree_batch_idx: list[str]) -> list[str]:
    return [f'{orginal_prefix}/{deeper_idx}' for deeper_idx, orginal_prefix in enumerate(reasoning_tree_batch_idx)]


def _get_vllm_inputs(prompts: DataProto, pad_token_id):
    batch_size = len(prompts)
    non_tensor_batch = prompts.non_tensor_batch
    if "raw_prompt_ids" not in non_tensor_batch:
        non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(pad_token_id, prompts.batch["input_ids"][i]) for i in range(batch_size)], dtype=object)

    if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
        raise RuntimeError("vllm sharding manager is not work properly.")

    vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]
    for input_data in vllm_inputs:
        if isinstance(input_data["prompt_token_ids"], np.ndarray):
            input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
        elif not isinstance(input_data["prompt_token_ids"], list):
            raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

    return vllm_inputs


def _get_root_node(reasoning_tree_batch_idx: str) -> int:
    return int(reasoning_tree_batch_idx.split('/')[0])


# def _is_answer_start(response_ids: list[int], answer_start_tag_tokens: list[int]) -> bool:
#     return any(response_ids[i:i+len(answer_start_tag_tokens)] == answer_start_tag_tokens for i in range(len(response_ids) - len(answer_start_tag_tokens) + 1))

def _is_contain_subtoken(token_list: list[int], subtoken_list: list[int]) -> bool:
    assert len(subtoken_list) <= len(token_list), "subtoken_list should be a subset of token_list"
    return any(token_list[i:i+len(subtoken_list)] == subtoken_list for i in range(len(token_list) - len(subtoken_list) + 1))

# 找到 subtoken_list 在 token_list 中的起始位置
def _find_subtokens_idx(token_list: list[int], subtoken_list: list[int]) -> int:
    location = -1
    for i in range(len(token_list) - len(subtoken_list) + 1):
        if token_list[i:i+len(subtoken_list)] == subtoken_list:
            location = i
            break
    return location

def _find_all_subtokens_idx(token_list: list[int], subtoken_list: list[int]) -> list[int]:
    locations = []
    i = 0
    while i < len(token_list):
        if token_list[i:i+len(subtoken_list)] == subtoken_list:
            locations.append(i)
            i += len(subtoken_list)
        else:
            i += 1
    return locations

# 找到最后一个 subtoken_list 在 token_list 中的起始位置，
def _find_subtokens_idx_reverse(token_list: list[int], subtoken_list: list[int]) -> int:
    location = -1
    for i in range(len(token_list) - len(subtoken_list), 0, -1):
        if token_list[i:i+len(subtoken_list)] == subtoken_list:
            location = i
            break
    return location


def _get_generated_logprobs_from_vllm_output(vllm_output) -> list[list[float]]:
    # token_ids = vllm_output.token_ids
    # token_logprobs_list = vllm_output.logprobs # This is a list of dicts, one per token
    # assert token_logprobs_list
    # generated_logprobs = []    
    # for token_id, parallel_logprobs in zip(token_ids, token_logprobs_list):
    #     generated_logprobs.append(parallel_logprobs[token_id].logprob)
    # return generated_logprobs
    
    one_sequence_logprobs = [] # 最后是一个 list[list[float]]，每个 list[float] 是每个时间 t 的 logprob
    for token_logprob_dict in vllm_output.logprobs:
        current_t_logprobs = []
        for token_id in token_logprob_dict:
            current_t_logprobs.append(token_logprob_dict[token_id].logprob)
        one_sequence_logprobs.append(current_t_logprobs)      
    return one_sequence_logprobs

def weight_to_discrete_allocate(weights, total):
    """
    每个序列至少分到 1。要求 total >= len(weights)
    """
    w = np.asarray(weights, dtype=float)
    N = len(w)

    if total < N:
        raise ValueError(f"资源 {total} 少于序列数 {N}，无法保证每个 >=1")

    # 先保底 1 份
    alloc = np.ones(N, dtype=int)
    rest  = total - N
    if rest == 0:
        return alloc.tolist()

    # 归一化权重（可以把 0 权重全置为极小正数防止除零）
    if w.sum() == 0:
        raise ValueError("权重全为 0，无法按比例分配")
    w = w / w.sum()

    # 最大余数法分配剩余 rest 份
    raw    = w * rest
    base   = np.floor(raw).astype(int)
    remain = rest - base.sum()

    if remain:
        frac   = raw - base
        order  = np.argsort(-frac)
        base[order[:remain]] += 1

    alloc += base
    return alloc.tolist()

# 定义一个数据类型用来保存推理树的结构
# 里面可以保存 tree_idx, token_ids，logprobs 等等
class SampleStatus(Enum):
    INITIAL = "initial"
    TO_INFER = "to_infer"
    FINISH_NEXT_INFER = "finish_next_infer"
    FINISHED = "finished"
    
class FinishedReason(Enum):
    FINISHED = "finished"
    UNCLOSED_ANSWER = "finished_with_unclosed_answer"
    EARLY_STOPPED_BY_EOS = "early_stopped_by_eos"
    MAX_INFER_STEPS = "finished_with_max_infer_steps"
    MAX_OUTPUT_TOKENS = "finished_with_max_output_tokens"
    REPETITION_STOP = "early_stopped_by_repetition"
    UNFINISHED = "unfinished" # 优先级最低
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return FINISHED_REASON_PRIORITY[self.value] < FINISHED_REASON_PRIORITY[other.value]
        return NotImplemented


FINISHED_REASON_PRIORITY = {
    "finished": 7,
    "finished_with_unclosed_answer": 6,
    "early_stopped_by_eos": 5,
    "finished_with_max_infer_steps": 4,
    "finished_with_max_output_tokens": 3,
    "early_stopped_by_repetition": 2,
    "unfinished": 1,
}

LOGPROG_PLACEHOLDER = -1e5

# https://paste.ubuntu.com/p/J7WXFpBXWx/
def _has_repetition(s: str, rep_length_thresh: int = 32, rep_count_thresh: int = 8):
    if not s or len(s) < rep_length_thresh:
        return False, ""
    subsequence_count = {}
    length = rep_length_thresh
    for i in range(len(s) - length + 1):
        subseq = s[i : i + length]
        subsequence_count[subseq] = subsequence_count.get(subseq, 0) + 1
        if subsequence_count[subseq] >= rep_count_thresh:
            return (
                True,
                "-" * 16
                + "Repitition Check"
                + "-" * 16
                + f"\nRepeated {subsequence_count[subseq]} times: {subseq}",
            )
    return False, ""


@dataclass
class DataSampleTree:
    """
    tree_idx 定义为 str，方便后续的扩展。
    具体含义举例 "0/1-0/2-9/4-1":
    * 表示根节点 0，然后 1-0 指的是在第 1 批次做 infer，且是 inference batch 中的第 0 个样本，
    * 然后 4-1 指的是在第 4 批次做 infer，且是 inference batch 中的第 1 个样本。
    
    token_count_per_step: 记录每个 inference step 生成的 token 数量
    """
    tree_idx: str
    init_input_len: int # 设置为不可修改？
    
    input_ids: list[int] # 这里可能包含初始输入和中间生成的输出，用于构造 vLLM 的输入
    full_response_token_ids: list[int] = field(default_factory=list) # 这里只包含最终的输出，用于计算 logprobs
    # response_str: str = ""
    logprobs: list[list[float]] = field(default_factory=list) # 两层 list，每个 list[float] 是每个时间 t 的 logprobs

    cumulative_logprobs: list[float] = field(default_factory=list) # 指的是
    
    # is_finished: bool = False
    finished_reason: FinishedReason = FinishedReason.UNFINISHED
    # 可以按照 finished_reason 的优先级排序, 举例：sorted_samples = sorted(samples, key=lambda s: s.finished_reason.priority)
    status: SampleStatus = SampleStatus.INITIAL
    token_count_per_step: List[int] = field(default_factory=list)  # 记录每一步生成的 token 数量
    
    def __post_init__(self):
        # 确保类型安全
        if not isinstance(self.finished_reason, FinishedReason):
            raise TypeError("finished_reason must be a FinishedReason enum")
        
    @property
    def root_node(self):
        return int(self.tree_idx.split('/')[0])
    @property
    def depth(self):
        return len(self.tree_idx.split('/')) - 1
    @property
    def input_len(self):
        return len(self.input_ids)      
    @property
    def output_len(self):
        return len(self.full_response_token_ids)
    @property
    def init_input_ids(self):
        return self.input_ids[:self.init_input_len]
    @property
    def actual_response_len(self):
        return len(self.full_response_token_ids)
    
    def extend_input_ids(self, token_ids: list[int]):
        self.input_ids.extend(token_ids)
        
    def extend_logprobs(self, logprobs: list[float]):
        self.logprobs.extend(logprobs)
        
    def extend_full_response_token_ids(self, token_ids: list[int]):
        self.full_response_token_ids.extend(token_ids)

    def extend_response(self, 
            response_id: list[int],
            # response_str: str,
            response_logprobs: list[float]):
        self.extend_full_response_token_ids(response_id)
        self.extend_logprobs(response_logprobs)
        self.token_count_per_step.append(len(response_id))  # 新增：记录这一步生成了多少 token
    def truncate_response(self, truncate_len: int):
        self.full_response_token_ids = self.full_response_token_ids[:-truncate_len]
        # self.response_str = self.response_str[:truncate_str_len]
        self.logprobs = self.logprobs[:-truncate_len]

    def truncate_tree_idx(self, truncate_len: int):
        assert truncate_len >= 0
        if truncate_len == 0:
            # do nothing
            pass    
        else:
            all_idxs = self.tree_idx.split('/')
            root_node = all_idxs[0]
            all_idxs = all_idxs[1:]
            new_tree_idx = "/".join([root_node] + all_idxs[:-truncate_len])            
            self.tree_idx = new_tree_idx
    
    def __post_init__(self):
        # 确保类型安全
        if not isinstance(self.finished_reason, FinishedReason):
            raise TypeError("finished_reason must be a FinishedReason enum")
   

# 按优先级对 finished samples 排序，但引入随机性
def _gumbel_topk_permutation(samples: List[DataSampleTree]):
    weights = [FINISHED_REASON_PRIORITY[sample.finished_reason.value] for sample in samples]
    def sample_gumbel():
        u = random.random()
        return -math.log(-math.log(u))
    
    # 为每个元素生成扰动后的打分
    scored = []
    for item, w in zip(samples, weights):
        score = math.log(w) + sample_gumbel()
        scored.append((score, item))
    
    # 按 score 降序排列，抽取元素
    scored.sort(reverse=True)
    return [item for _, item in scored]

# 按优先级对任意 list 排序，但引入随机性，默认权重越大越优先
def _gumbel_topk_permutation_list(samples: List[Any], weights: List[float], reverse=False):
    def sample_gumbel():
        u = random.random()
        return -math.log(-math.log(u))
    eps = 1e-9
    n = len(weights)
    # 1. 归一化权重，使得 sum(norm_weights) == 1
    total = sum(weights)
    if total > 0:
        norm_weights = [w / total for w in weights]
    else:
        # 若所有权重都为 0，则退化为均等权重
        norm_weights = [1.0 / n] * n
    # 2. 计算每个元素的 Gumbel 分数
    scored = []
    for item, w in zip(samples, norm_weights):
        base = math.log(w + eps)
        if reverse:
            base = -base
        score = base + sample_gumbel()
        scored.append((score, item))
    # 3. 按 score 从大到小排序并返回对应的样本顺序
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]
     
def get_vllm_inputs_from_samples(samples: list[DataSampleTree]) -> list[dict]:
    vllm_inputs = []
    for sample in samples:
        vllm_inputs.append({
            "prompt_token_ids": sample.input_ids,
        })
    return vllm_inputs

def _increment_tree_idx_depth(
    samples: list[DataSampleTree],
    next_infer_step: int,
    ) -> list[DataSampleTree]:
    """
    根据 next_infer_step 和 infer_batch_idx 更新 samples 的 tree_idx（主要是增加一层）
    """
    for infer_batch_idx, sample in enumerate(samples):
        sample.tree_idx = sample.tree_idx + "/" + f"{next_infer_step}-{infer_batch_idx}"
    return samples

import math
def compute_sequence_entropy_from_topk(topk_logprobs_list, skip_placeholder: bool = True):
    """ 估算 token-level entropy（基于 top-k logprobs）"""
    entropies = []
    for logprobs in topk_logprobs_list:
        if skip_placeholder and (logprobs[0] - LOGPROG_PLACEHOLDER) < 1e-9:
            continue
        probs = [math.exp(lp) for lp in logprobs]
        norm = sum(probs)
        probs = [p / (norm + 1e-9) for p in probs]
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        entropies.append(entropy)
    return np.mean(entropies)

import re
def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None


class vLLMTest:
    def __init__(
        self, 
        model_path: str, 
        tensor_parallel_size: int = 1,
        model_type: str = "step_model",
        gpu_util: float = 0.6,
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model_type = model_type
        
        if model_type == "step_model":
            # pre-defined special tags (in list[int])
            self.step_start_tag_tokens = self.tokenizer.encode("<step>\n", add_special_tokens=False)
            self.step_end_tag_tokens = self.tokenizer.encode("</step>\n", add_special_tokens=False)
            self.answer_start_tag_tokens = self.tokenizer.encode("<answer>\n", add_special_tokens=False)
            self.single_newline_token = self.tokenizer.encode("\n", add_special_tokens=False)
            self.answer_end_tag_tokens = self.tokenizer.encode("</answer>", add_special_tokens=False) # 不在前面加 \n，因为前面的 \n 可能会被其他 token 吃进 BPE
        elif model_type == "base_model":
            # pre-defined special tags (in list[int])
            self.answer_start_tag_tokens = self.tokenizer.encode("\n\nIn conclusion, the final answer is", add_special_tokens=False)
            
        # print("pre-defined special tags:", type(self.step_start_tag_tokens))
        # print(self.step_start_tag_tokens)
        # print(self.step_end_tag_tokens)
        print(self.answer_start_tag_tokens, self.tokenizer.decode(self.answer_start_tag_tokens))
        # print(self.answer_end_tag_tokens)
        
        self.max_model_len = 4096 
        self.max_input_len = 1024
        self.max_output_len = 3072
        
        self.inference_engine = LLM(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            enforce_eager=True, # vLLMv1 false
            gpu_memory_utilization=gpu_util,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        # print("stop token ids:")
        # print(self.tokenizer.eos_token_id)
        # print(self.tokenizer.encode("<|im_end|>", add_special_tokens=False))
        # print(self.tokenizer.pad_token_id)
        self.sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=0.95,
            top_k=-1,
            max_tokens=self.max_output_len,
            detokenize=True,
            # stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
            # stop=["\n</answer>", "</answer>","<|im_end|>"],
            # stop=["<|im_end|>"],
            include_stop_str_in_output=True,
        )
        
        

    def prepare_input(self, prompts: list[str]) -> DataProto:
        batch_dict = {
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [],
        }
        for prompt in prompts:
            if self.model_type == "step_model":
                chat = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\\\boxed{}. The reasoning steps should be enclosed with <step> </step> and the boxed answer in <answer> </answer> tags."},
                    {"role": "user", "content": "Question: " + prompt}
                ]
            elif self.model_type == "base_model":
                chat = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\\\boxed{}."},
                    {"role": "user", "content": prompt}
                ]
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, 
                add_generation_prompt=True, 
                tokenize=False
            )
            model_inputs = self.tokenizer(prompt_with_chat_template, return_tensors="pt")
            input_ids = model_inputs.pop('input_ids')
            attention_mask = model_inputs.pop('attention_mask')
            input_ids, attention_mask = verl_F.postprocess_data(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        max_length=self.max_input_len,
                                                        pad_token_id=self.tokenizer.pad_token_id,
                                                        left_pad=True,
                                                        truncation="left") 
            
            # Compute position ids from attention mask
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict['input_ids'].append(input_ids[0])
            batch_dict['attention_mask'].append(attention_mask[0])
            batch_dict['position_ids'].append(position_ids[0])
        batch_dict['input_ids'] = torch.stack(batch_dict['input_ids'],dim=0)
        batch_dict['attention_mask'] = torch.stack(batch_dict['attention_mask'],dim=0)
        batch_dict['position_ids'] = torch.stack(batch_dict['position_ids'],dim=0)
        
        input_batch = DataProto.from_single_dict(batch_dict)
        return input_batch

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
            
            
    def generate_sequence_sequential(self, prompts: DataProto, **kwargs) -> DataProto:
        # repeating the inputs rather use sampling-n 
        if kwargs.get("n", 1) > 1:
            prompts = prompts.repeat(kwargs["n"], interleave=True)
            # If prompts are repeated, vLLM should generate 1 sequence for each repeated prompt.
            kwargs["n"] = 1
        
        calculate_entropy = kwargs.get("calculate_entropy", False)
        if calculate_entropy:
            assert kwargs.get("logprobs", -1) >= 0, "n_return_logprobs should be greater than 0 if calculate_entropy is true"

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info["eos_token_id"]
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = idx.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.tokenizer.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")
        
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params, # self.sampling_params is updated by context manager
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

            response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=self.max_output_len).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.max_output_len).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            if calculate_entropy:
                all_sequence_logprobs = [] # Store logprobs for later processing
                for output_item in outputs: # output_item is RequestOutput
                    one_sequence_logprobs = _get_generated_logprobs_from_vllm_output(output_item.outputs[0])
                    all_sequence_logprobs.append(one_sequence_logprobs)
                    
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

        if calculate_entropy:
            # entropies_tensor = torch.tensor(sequence_entropies, dtype=torch.float32, device=idx.device)
            # if len(entropies_tensor) == batch_size: # current_batch_size is idx.shape[0]
                # batch["response_entropy"] = entropies_tensor
            assert len(all_sequence_logprobs) == batch_size, f"Logprob extraction produced {len(all_sequence_logprobs)} items, but current batch size is {batch_size}. Entropy will not be added."
            non_tensor_batch["response_logprobs"] = np.array(all_sequence_logprobs, dtype=object)
            
        # breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        
    def generate_sequences_tree_deepth_first(
        self, 
        prompts: DataProto, 
        fixed_step_width: int = 2,
        max_token_per_step: int = 512, 
        max_depth: int = 6,
        max_width: int = 8, # ~= rollout n trajectory
        force_answer_remaining_token: int = 128,
        # force_answer_threshold: int = 2,
        **kwargs) -> DataProto:
        
        # assert max_depth * max_token_per_step <= self.max_output_len, f"only supports max_depth * max_token_per_step (current: {max_depth * max_token_per_step}) <= max_output_len (current: {self.max_output_len}) ATM"
        
        calculate_entropy = kwargs.get("calculate_entropy", False)
        if calculate_entropy:
            assert kwargs.get("logprobs", -1) >= 0, "n_return_logprobs should be greater than 0 if calculate_entropy is true"
        
        start_time = time.time()
        
        # TODO: implement the tree search by max token
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info["eos_token_id"]
        eos_token_id = self.tokenizer.eos_token_id
        
        batch_size = idx.size(0)
        
        n_target_trajectory = len(prompts) * max_width
        non_tensor_batch = prompts.non_tensor_batch
        vllm_inputs = _get_vllm_inputs(prompts, self.tokenizer.pad_token_id)

        
        init_input_ids_by_root_node = {root_idx: vllm_input['prompt_token_ids'] for root_idx, vllm_input in enumerate(vllm_inputs)}
        # input_len_by_root_node = {root_idx: len(vllm_input['prompt_token_ids']) for root_idx, vllm_input in enumerate(vllm_inputs)}
        width_counter = {root_idx: 0 for root_idx in range(len(vllm_inputs))} # record number offinished samples for each root node
        finished_samples = {root_idx: [] for root_idx in range(len(vllm_inputs))} # store the tuples of (finished response ids, reasoning tree batch idx, finished_status)
        
        
        # 构造第一批输入
        samples_to_infer = []
        for root_idx in range(len(vllm_inputs)):
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
        next_infer_step = 1
        samples_to_infer = _repeat_list_interleave(samples_to_infer, fixed_step_width)
        samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step=next_infer_step)
        step_start_times = [start_time]  # 记录每个 step 开始的时间
        step_efficiency_metrics = []
        
        while len(samples_to_infer) > 0:
            assert kwargs.get("n", 1) == 1, "n must be 1 for tree search, use repeat_interleave to generate multiple samples"
            kwargs["max_tokens"] = max_token_per_step
            
            # # debug：decode 并检查输入中是否存在 "</answer>"
            # for sample in samples_to_infer:
            #     input_str_after_init = self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False)
            #     if input_str_after_init.find("</answer>") >= 0:
            #         print(f"find </answer> in input_str: {input_str_after_init}")
            #         print(f"reasoning tree batch idx: {sample.tree_idx}")
            #         print("--------------------------------")
            
            # if isinstance(vllm_inputs[0], list):
            #     vllm_inputs = [{"prompt_token_ids": vllm_input} for vllm_input in vllm_inputs]
            vllm_inputs = get_vllm_inputs_from_samples(samples_to_infer)
            print(f"vllm_inputs to infer: {len(vllm_inputs)}")
            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                current_step_end_time = time.time()
                step_duration = current_step_end_time - step_start_times[-1]
                step_start_times.append(current_step_end_time)
                step_efficiency_metrics.append({
                    "step_duration": step_duration,
                    "max_token_per_step": kwargs["max_tokens"],
                    "n_inputs": len(vllm_inputs),
                    "average_input_tokens": np.mean([len(list(sample.input_ids)) for sample in samples_to_infer]),
                    "n_outputs": len(outputs),
                    "average_output_tokens": np.mean([len(list(output.outputs[0].token_ids)) for output in outputs]),
                })
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
                # sample.token_count_per_step.append(len(response_id)) # 记录step长度
                answer_start_idx = _find_subtokens_idx(response_id, self.answer_start_tag_tokens)
                answer_end_idx = _find_subtokens_idx_reverse(response_id, self.answer_end_tag_tokens)
                last_eos_idx = _find_subtokens_idx_reverse(response_id, [self.tokenizer.eos_token_id])
                

                actual_response = sample.full_response_token_ids
                acutal_response_len = sample.actual_response_len
                
                # debug: sanity check
                answer_start_str_idx = response_str.find("<answer>")
                answer_end_str_idx = response_str.find("</answer>")
                # eos_str_idx = response_str.find("<|im_end|>")
                # if  answer_end_str_idx >= 0 and eos_str_idx >= 0:
                #     assert answer_end_str_idx < eos_str_idx, f"answer end str idx {answer_end_str_idx} should be less than eos str idx {eos_str_idx}, response:\n{response_str}"
                #     assert len("<|im_end|>") + eos_str_idx == len(response_str), f"eos str idx {eos_str_idx} should be at the end of the response, response:\n{response_str}"
                
                # 上一轮强行要求作答的 query，不论有没有作答都要结束了
                if sample.status == SampleStatus.FINISH_NEXT_INFER:
                    # print(f"answer start found at input, stop generation, response:\n{response_str}")
                    if answer_end_str_idx >= 0:
                        sample.finished_reason = FinishedReason.FINISHED
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                    else:
                        sample.finished_reason = FinishedReason.UNCLOSED_ANSWER
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                # 如果 response 中存在 </answer>，则认为这条路径已经结束
                elif answer_end_str_idx >=0:
                    sample.finished_reason = FinishedReason.FINISHED
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue

                elif answer_start_str_idx >= 0:
                    if last_eos_idx > answer_start_idx:
                        # 如果 eos token 在 answer start token 之后，虽然没有闭合 answer，但认为这条路径已经结束
                        # print("no paired answer end found, but eos token found")
                        sample.finished_reason = FinishedReason.UNCLOSED_ANSWER
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                        width_counter[sample.root_node] += 1
                        finished_samples_this_step_by_root_node[sample.root_node] += 1
                    else:
                        # 如果 answer 没有闭合，则认为这条路径没有结束，继续完成作答深度搜索（半步？）
                        sample.status = SampleStatus.FINISH_NEXT_INFER
                        sample.input_ids = sample.input_ids + response_id
                        # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                        #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                        samples_to_infer.append(sample)
                    continue


                # 如果没有 answer，则认为这条路径没有结束，判断是否要继续 rollout
                # 如果实际 response 长度 >= max_output_len，则认为这条路径已经结束
                elif acutal_response_len >= self.max_output_len:
                    print(f"{sample.tree_idx}: actual response length >= max_output_len, stop")
                    sample.finished_reason = FinishedReason.MAX_OUTPUT_TOKENS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                
                # 如果剩余 token budget <= force_answer_remaining_token，则开始强行输出回答
                # TODO：这里需要考虑 logprobs 的 truncate，而且由于会强行 append 新的 token，需要考虑这些 token 的 logprobs 怎么处理？
                elif self.max_output_len - acutal_response_len <= force_answer_remaining_token:                  
                    last_step_start = _find_subtokens_idx_reverse(actual_response, self.step_start_tag_tokens)
                    last_step_end = _find_subtokens_idx_reverse(actual_response, self.step_end_tag_tokens)
                    if last_step_end < 0 and last_step_start < 0:
                        print(f"{sample.tree_idx}: no step end or step start token found")
                        # Bad case，没有 step end 和 step start token，直接去掉后面的 token，做强行补齐
                        # 能够这么做的原因是这种 bad prefix path 按理说不会产生正确答案（需要二次确认）
                        truncate_len = force_answer_remaining_token - (self.max_output_len - acutal_response_len)
                        sample.input_ids = sample.init_input_ids + actual_response[:-truncate_len] + self.answer_start_tag_tokens
                        # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                        #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                        sample.truncate_response(truncate_len=truncate_len) # 这里需要切掉 response_logprobs 和 response_ids
                        sample.extend_response(
                            response_id=self.answer_start_tag_tokens, 
                            response_logprobs=[[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))]*len(self.answer_start_tag_tokens)
                        )
                    elif last_step_end > last_step_start:
                        # 最后一个 step 完整： ...<step>...</step>...
                        # 但可能存在格式之外的内容，先忽略不计，此时切掉 last </step> 后面的内容
                        truncate_len = last_step_end+len(self.step_end_tag_tokens)
                        sample.input_ids = sample.init_input_ids + actual_response[:truncate_len] + self.answer_start_tag_tokens
                        # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                        #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                        sample.truncate_response(truncate_len=truncate_len) # 这里需要切掉 response_logprobs 和 response_ids
                        sample.extend_response(
                            response_id=self.answer_start_tag_tokens, 
                            response_logprobs=[[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))]*len(self.answer_start_tag_tokens)
                        )
                    else:
                        # 存在未完成 step： ...</step><step>...
                        # 直接补充 </step> 并开始作答，不做 truncate
                        sample.input_ids = sample.init_input_ids + actual_response + self.step_end_tag_tokens + self.answer_start_tag_tokens
                        # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                        #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                        sample.extend_response(
                            response_id=self.step_end_tag_tokens + self.answer_start_tag_tokens, 
                            response_logprobs=[[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))]*len(self.step_end_tag_tokens + self.answer_start_tag_tokens)
                        )
                    samples_to_infer.append(sample)
                    continue
                    
                elif sample.depth + 1 >= max_depth:
                    print(f"{sample.tree_idx}: depth {sample.depth} about to reach max_depth, force to answer")
                    sample.input_ids = sample.input_ids + response_id + self.answer_start_tag_tokens
                    # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                    #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                    sample.status = SampleStatus.FINISH_NEXT_INFER
                    sample.extend_response(
                        response_id=self.answer_start_tag_tokens, 
                        response_logprobs=[[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))]*len(self.answer_start_tag_tokens)
                    )
                    samples_to_infer.append(sample)
                    continue

                else:
                    # if response_str.find("</answer>") >= 0 or self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                    #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                    # 如果还有足够的 token budget，则继续 rollout
                    assert sample.status == SampleStatus.TO_INFER, f"sample {sample.tree_idx} should be TO_INFER, but is {sample.status}"
                    samples_to_go_deeper[sample.root_node].append(copy.deepcopy(sample))
            
            root_to_infer_count = defaultdict(int)
            for sample in samples_to_infer:
                root_to_infer_count[sample.root_node] += 1
                
            if samples_to_go_deeper:
                print(f"roots to go deeper after step {next_infer_step}: {len(samples_to_go_deeper)}")
                for root_node in samples_to_go_deeper:
                    remaining_width_budget = max_width - width_counter[root_node] - root_to_infer_count.get(root_node, 0)
                    extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * fixed_step_width
                    
                    total_divergence = min([remaining_width_budget, extra_allow_divergence + fixed_step_width * len(samples_to_go_deeper[root_node])])
                    
                    # print(f"root_node: {root_node}, finished: {width_counter[root_node]}, to go deeper: {len(samples_to_go_deeper[root_node])}")
                    print(f"root {root_node}; already finished {width_counter[root_node]}; number of samples finishing in this step: {finished_samples_this_step_by_root_node[root_node]}; remaining width budget: {remaining_width_budget}; path to go deeper: {len(samples_to_go_deeper[root_node])}; divergence for next round {total_divergence=}")
                    average_divergence = total_divergence // len(samples_to_go_deeper[root_node])
                    remainder_divergence = total_divergence % len(samples_to_go_deeper[root_node])

                    # 将 total_divergence 的预算，均匀分配给 samples_to_go_deeper[root_node] 中的每个样本
                    for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                        if path_idx + 1 == len(samples_to_go_deeper[root_node]):
                            n_rollout = average_divergence + remainder_divergence
                        else:
                            n_rollout = average_divergence
                        for _ in range(n_rollout):
                            new_sample = copy.deepcopy(sample)
                            new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            # if self.tokenizer.decode(new_sample.input_ids[new_sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                            #     print(f"[ERROR] find additional</answer> in input_ids: {new_sample.input_ids}")
                            samples_to_infer.append(new_sample)
                            root_to_infer_count[root_node] += 1
                        
            # 不做 fallback？
            for root_idx in width_counter: 
                remaining_width_budget = max_width - (root_to_infer_count.get(root_idx, 0) + width_counter[root_idx])
                max_divergence_nodes = remaining_width_budget // fixed_step_width
                # 如果这个 query 没有深度搜索的 path，且还有宽度预算，则从 finished_samples 中取样
                if root_idx not in samples_to_go_deeper and remaining_width_budget > 0:
                    print(f"root_idx {root_idx} has remaining width budget, but no go-deeper path, build {remaining_width_budget} samples to infer")
                    # finished_samples[root_idx].sort(key=lambda x: int(x[2].split("_")[0]))
                    finished_samples[root_idx] = sorted(finished_samples[root_idx], key=lambda s: s.finished_reason)
                    for finished_sample in finished_samples[root_idx]:
                        new_sample = copy.deepcopy(finished_sample)
                        finished_output_ids = new_sample.full_response_token_ids
                        all_step_start_idx = _find_all_subtokens_idx(finished_output_ids, self.step_start_tag_tokens)
                        
                        # randomly choose one location to fallback
                        if all_step_start_idx:
                            fallback_idx = np.random.choice(all_step_start_idx)
                            truncate_len = len(new_sample.full_response_token_ids) - fallback_idx
                        else:
                            print(f"sample {new_sample.tree_idx} could not find step start token, fallback to random position.\nexisting response: {new_sample.response_str}")
                            fallback_idx = np.random.randint(0, new_sample.output_len)
                            truncate_len = new_sample.output_len - fallback_idx
                        new_sample.truncate_response(truncate_len=truncate_len)                        
                        new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            
                        new_sample.status = SampleStatus.TO_INFER
                        
                        if remaining_width_budget >= fixed_step_width:
                            for _ in range(fixed_step_width):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                            remaining_width_budget -= fixed_step_width
                        else:
                            for _ in range(remaining_width_budget):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                            remaining_width_budget = 0
                
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
        print(f"time cost: {end_time - start_time} seconds")
        print(f"total number of trajectory: {total_n_trajectory}")
        print(f"average time per trajectory: {(end_time - start_time) / total_n_trajectory} seconds")
        # assert sum(len(finished_samples[root_idx]) for root_idx in finished_samples) == n_target_trajectory, "finished_samples should be equal to n_target_trajectory"

        # 对齐原有输出格式
        response = []
        tree_indices = []
        response_root_indices = []
        for root_idx in range(len(prompts)):
            for sample in finished_samples[root_idx]:
                response.append(sample.full_response_token_ids)
                response_root_indices.append(root_idx) # 保存每一条 response 对应的 root idx，因为后面要用来做 repeat（数量不一致的情况不能直接 repeat interleave。）
                tree_indices.append(copy.copy(sample.tree_idx))

        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=self.max_output_len)
        if max_width > 1:
            idx = _repeat_interleave(idx, max_width)
            attention_mask = _repeat_interleave(attention_mask, max_width)
            position_ids = _repeat_interleave(position_ids, max_width)
            batch_size = batch_size * max_width
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

    def generate_sequences_tree_deepth_first_vanilla_mode(
        self, 
        prompts: DataProto, 
        fixed_step_width: int = 2,
        random_first_div_max: int = -1,
        max_token_per_step: int = 512, 
        max_depth: int = 6,
        max_width: int = 8, # ~= rollout n trajectory
        force_answer_remaining_token: int = 128,
        # force_answer_threshold: int = 2,
        fallback_policy: str = "random",
        divergence_policy: str = "fixed_avg",
        divergence_budget_control: str = "by_final_trajectory", # or "by_infer_step_token_budget"
        logprob_div_temperature: float = 1.0,
        cal_prob_block_size: int = 128,
        # flexible_max_width: bool = False,
        **kwargs) -> DataProto:
        
        """
        implement tree inference for model without SFT cold-start
        """
        
        # assert max_depth * max_token_per_step <= self.max_output_len, f"only supports max_depth * max_token_per_step (current: {max_depth * max_token_per_step}) <= max_output_len (current: {self.max_output_len}) ATM"
        
        calculate_entropy = kwargs.get("calculate_entropy", False)
        if calculate_entropy:
            assert kwargs.get("logprobs", -1) >= 0, "n_return_logprobs should be greater than 0 if calculate_entropy is true"
        
        start_time = time.time()
        
        # TODO: implement the tree search by max token
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info["eos_token_id"]
        eos_token_id = self.tokenizer.eos_token_id
        
        batch_size = idx.size(0)
        
        meta_info = {} # store the meta_info to return
        
        n_target_trajectory = len(prompts) * max_width
        non_tensor_batch = prompts.non_tensor_batch
        vllm_inputs = _get_vllm_inputs(prompts, self.tokenizer.pad_token_id)

        
        init_input_ids_by_root_node = {root_idx: vllm_input['prompt_token_ids'] for root_idx, vllm_input in enumerate(vllm_inputs)}
        # input_len_by_root_node = {root_idx: len(vllm_input['prompt_token_ids']) for root_idx, vllm_input in enumerate(vllm_inputs)}
        width_counter = {root_idx: 0 for root_idx in range(len(vllm_inputs))} # record number offinished samples for each root node
        finished_samples = {root_idx: [] for root_idx in range(len(vllm_inputs))} # store the tuples of (finished response ids, reasoning tree batch idx, finished_status)
        
        fallback_counter = defaultdict(int) # 记录 fallback 的次数
        n_finished_traj_at_first_fallback = {}
        
        # 构造第一批输入
        samples_to_infer = []
        for root_idx in range(len(vllm_inputs)):
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
        if divergence_budget_control == "by_infer_step_token_budget":
            print(f"[WARNING] setting {divergence_budget_control=}, now use the {fixed_step_width=} as the total maximum inference token budget, and {max_token_per_step=} as the budget for vLLM params for each input.")
            first_step_div = fixed_step_width//max_token_per_step//len(samples_to_infer)
            if first_step_div <= 1:
                print(f"[WARNING] average first divergence {first_step_div=} <=1, please set a larger value of {fixed_step_width=}.")
            first_step_div = max(1, first_step_div) # at least one
            first_step_div = min(first_step_div, max_width)
            samples_to_infer = _repeat_list_interleave(samples_to_infer, first_step_div)
        elif random_first_div_max > fixed_step_width:
            assert random_first_div_max <= max_width, f"{random_first_div_max=} should be less than or equal to {max_width=}"
            duplicated_indices = []
            for sample in samples_to_infer:
                # 这里的 sample.tree_idx 是 str 类型
                first_step_div = np.random.randint(fixed_step_width, random_first_div_max+1) 
                duplicated_indices.extend([sample.root_node]*first_step_div)
            # 这里的 duplicated_indices 是 list[int]，每个元素是 sample.root_node 的索引
            samples_to_infer = _repeat_list_by_indices(samples_to_infer, duplicated_indices)
            print(f"first step rollout, duplicate {len(vllm_inputs)} samples to infer: {len(samples_to_infer)}")
        else:
            samples_to_infer = _repeat_list_interleave(samples_to_infer, fixed_step_width)
        
        next_infer_step = 1
        samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step=next_infer_step)
        step_start_times = [start_time]  # 记录每个 step 开始的时间
        step_efficiency_metrics = []
        while len(samples_to_infer) > 0:
            assert kwargs.get("n", 1) == 1, "n must be 1 for tree search, use repeat_interleave to generate multiple samples"
            kwargs["max_tokens"] = max_token_per_step
            
            # if isinstance(vllm_inputs[0], list):
            #     vllm_inputs = [{"prompt_token_ids": vllm_input} for vllm_input in vllm_inputs]
            vllm_inputs = get_vllm_inputs_from_samples(samples_to_infer)
            print(f"vllm_inputs to infer: {len(vllm_inputs)}")
            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                current_step_end_time = time.time()
                step_duration = current_step_end_time - step_start_times[-1]
                step_start_times.append(current_step_end_time)
                step_efficiency_metrics.append({
                    "step_duration": step_duration,
                    "max_token_per_step": kwargs["max_tokens"],
                    "n_inputs": len(vllm_inputs),
                    "average_input_tokens": float(np.mean([len(list(sample.input_ids)) for sample in samples_to_infer])),
                    "n_outputs": len(outputs),
                    "average_output_tokens": float(np.mean([len(list(output.outputs[0].token_ids)) for output in outputs])),
                })
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
                    
                answer_start_idx = _find_subtokens_idx(response_id, self.answer_start_tag_tokens)
                last_eos_idx = _find_subtokens_idx_reverse(response_id, [self.tokenizer.eos_token_id])
                
                boxed_answer = extract_last_boxed(response_str)

                actual_response = sample.full_response_token_ids
                acutal_response_len = sample.actual_response_len
                
                # 上一轮强行要求作答的 query，不论有没有作答都要结束了
                if sample.status == SampleStatus.FINISH_NEXT_INFER:
                    # print(f"answer start found at input, stop generation, response:\n{response_str}")
                    if boxed_answer is not None:
                        sample.finished_reason = FinishedReason.FINISHED
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                    else:
                        sample.finished_reason = FinishedReason.UNCLOSED_ANSWER
                        sample.status = SampleStatus.FINISHED
                        finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                # 如果 response 中存在完整 \\boxed{}
                elif boxed_answer is not None:
                    sample.finished_reason = FinishedReason.FINISHED
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                elif last_eos_idx >= 0:
                    sample.finished_reason = FinishedReason.EARLY_STOPPED_BY_EOS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue

                # 如果没有 answer，则认为这条路径没有结束，判断是否要继续 rollout
                # 如果实际 response 长度 >= max_output_len，则认为这条路径已经结束
                elif acutal_response_len >= self.max_output_len:
                    print(f"{sample.tree_idx}: actual response length >= max_output_len, stop")
                    if acutal_response_len > self.max_output_len:
                        truncate_len = acutal_response_len - self.max_output_len
                        sample.truncate_response(truncate_len=truncate_len)     
                    sample.finished_reason = FinishedReason.MAX_OUTPUT_TOKENS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                
                # # 如果剩余 token budget <= force_answer_remaining_token，则开始强行输出回答
                # # TODO：这里需要考虑 logprobs 的 truncate，而且由于会强行 append 新的 token，需要考虑这些 token 的 logprobs 怎么处理？
                # elif self.max_output_len - acutal_response_len <= force_answer_remaining_token:                  
                #     # base model 不需要考虑 step end 和 step start token，直接去掉后面的 token，做强行补齐
                #     truncate_len = force_answer_remaining_token - (self.max_output_len - acutal_response_len)
                #     sample.input_ids = sample.init_input_ids + actual_response[:-truncate_len] + self.answer_start_tag_tokens
                #     # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                #     #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                #     sample.truncate_response(truncate_len=truncate_len) # 这里需要切掉 response_logprobs 和 response_ids
                #     sample.extend_response(
                #         response_id=self.answer_start_tag_tokens, 
                #         response_logprobs=[[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))]*len(self.answer_start_tag_tokens)
                #     )
                #     samples_to_infer.append(sample)
                #     continue
                elif sample.depth >= max_depth:
                    print(f"{sample.tree_idx}: reach max_depth, stop")
                    sample.finished_reason = FinishedReason.MAX_INFER_STEPS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    width_counter[sample.root_node] += 1
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                # elif sample.depth + 1 >= max_depth:
                #     print(f"{sample.tree_idx}: depth {sample.depth} about to reach max_depth, force to answer")
                #     sample.input_ids = sample.input_ids + response_id + self.answer_start_tag_tokens
                #     # if self.tokenizer.decode(sample.input_ids[sample.init_input_len:], skip_special_tokens=False).count("</answer>") >= 1:
                #     #     print(f"[ERROR] find additional</answer> in input_ids: {sample.input_ids}")
                #     sample.status = SampleStatus.FINISH_NEXT_INFER
                #     sample.extend_response(
                #         response_id=self.answer_start_tag_tokens, 
                #         response_logprobs=[[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))]*len(self.answer_start_tag_tokens)
                #     )
                #     samples_to_infer.append(sample)
                #     continue
                res_has_repretition, repeated_substr = _has_repetition(response_str)
                if res_has_repretition:
                    print(f"{sample.tree_idx}: response contains repetition, stop, [response]:\n{response_str}")
                    print(repeated_substr)
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
            
            root_to_infer_count = defaultdict(int)
            for sample in samples_to_infer:
                root_to_infer_count[sample.root_node] += 1
                
            if samples_to_go_deeper:
                print(f"roots to go deeper after step {next_infer_step}: {len(samples_to_go_deeper)}")
                for root_node in samples_to_go_deeper:
                    if divergence_budget_control == "by_infer_step_token_budget":
                        # the budget is pre-defined by total number of output tokens in `fixed_step_width`, the upper bound is definied by the final trajectory numbers.
                        token_budget_per_query = fixed_step_width//len(samples_to_go_deeper.keys())
                        remaining_width_budget = max_width - width_counter[root_node] - root_to_infer_count.get(root_node, 0)
                        total_divergence = min(remaining_width_budget, token_budget_per_query//max_token_per_step)
                    elif divergence_budget_control == "by_final_trajectory":
                        # the budget is transferred from the finished trajectories, the upper bound is definied by the final trajectory numbers.
                        remaining_width_budget = max_width - width_counter[root_node] - root_to_infer_count.get(root_node, 0)
                        extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * fixed_step_width                    
                        total_divergence = min([remaining_width_budget, extra_allow_divergence + fixed_step_width * len(samples_to_go_deeper[root_node])])
                    else:
                        raise NotImplementedError(f"{divergence_budget_control=} not implemented")
                    
                    if divergence_policy == "fixed_avg":
                        print(f"root {root_node}; already finished {width_counter[root_node]}; number of samples finishing in this step: {finished_samples_this_step_by_root_node[root_node]}; remaining width budget: {remaining_width_budget}; path to go deeper: {len(samples_to_go_deeper[root_node])}; divergence for next round {total_divergence=}")
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
                    elif divergence_policy == "logprob_weighted_div":
                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                        cum_logprob = torch.tensor([lp.sum() for lp in log_probs])  # shape (N,)
                        # 先做一个 shift，防止极端负值下溢
                        shifted = -1.0 * (cum_logprob - cum_logprob.min())
                        weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                        divergence_budgets = weight_to_discrete_allocate(weights, total_divergence)
                        for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                            for _ in range(div_budget):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    elif divergence_policy == "inverse_logprob_weighted_div":
                        # 如果 logprob 越大，分岔权重越大
                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                        cum_logprob = torch.tensor([lp.sum() for lp in log_probs])  # shape (N,)
                        # 先做一个 shift，防止极端负值下溢
                        shifted = 1.0 * (cum_logprob - cum_logprob.min())
                        weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                        divergence_budgets = weight_to_discrete_allocate(weights, total_divergence)
                        for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                            for _ in range(div_budget):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    elif divergence_policy == "norm_logprob_weighted_div":
                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                        cum_logprob = torch.tensor([lp.mean() for lp in log_probs])  # shape (N,)
                        # 先做一个 shift，防止极端负值下溢
                        shifted = -1.0 * (cum_logprob - cum_logprob.min())
                        weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                        divergence_budgets = weight_to_discrete_allocate(weights, total_divergence)
                        for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                            for _ in range(div_budget):
                                new_sample = copy.deepcopy(sample)
                                new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                                samples_to_infer.append(new_sample)
                                root_to_infer_count[root_node] += 1
                    elif divergence_policy == "norm_inv_logprob_weighted_div":
                        remaining_width_budget = max_width - width_counter[root_node] - root_to_infer_count.get(root_node, 0)
                        extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * fixed_step_width                    
                        total_divergence = min([remaining_width_budget, extra_allow_divergence + fixed_step_width * len(samples_to_go_deeper[root_node])])

                        log_probs = []                # (N, T_i) 列表或张量
                        for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                            log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                        cum_logprob = torch.tensor([lp.mean() for lp in log_probs])  # shape (N,)
                        # 先做一个 shift，防止极端负值下溢
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
                remaining_width_budget = max_width - (root_to_infer_count.get(root_idx, 0) + width_counter[root_idx])
                # max_divergence_nodes = remaining_width_budget // fixed_step_width
                # 如果这个 query 没有深度搜索的 path，且还有宽度预算，则从 finished_samples 中取样
                if root_idx not in samples_to_go_deeper and remaining_width_budget > 0:
                    print(f"root_idx {root_idx} has remaining width budget, but no go-deeper path, build {remaining_width_budget} samples to infer")
                    
                    # 记录第一次发生 fallback 的时候有多少条正常跑完的 trajectory
                    if root_idx not in n_finished_traj_at_first_fallback:
                        n_finished_traj_at_first_fallback[root_idx] = root_to_infer_count.get(root_idx, 0) + width_counter[root_idx]
                        
                    # finished_samples[root_idx].sort(key=lambda x: int(x[2].split("_")[0]))
                    finished_samples[root_idx] = sorted(finished_samples[root_idx], key=lambda s: s.finished_reason)
                    for finished_sample in finished_samples[root_idx]:
                        new_sample = copy.deepcopy(finished_sample)
                        
                        if fallback_policy == "random":
                            # 随机 fallback 到某个位置
                            fallback_idx = np.random.randint(0, 1+max(0, (new_sample.output_len-1))//max_token_per_step) * max_token_per_step
                        elif fallback_policy == "min_logprob_token":
                            new_sample_response_logprobs = new_sample.logprobs
                            new_sample_response_logprobs = [x[0] for x in new_sample_response_logprobs]
                            fallback_idx = np.argmin(new_sample_response_logprobs)
                        elif fallback_policy == "block":
                            new_sample_response_logprobs = new_sample.logprobs
                            new_sample_response_logprobs = [x[0] for x in new_sample_response_logprobs]
                            sub_block_logprobs_list = [new_sample_response_logprobs[i:i+cal_prob_block_size] for i in range(0, len(new_sample_response_logprobs), cal_prob_block_size)]
                            logprobs_mean_list = [np.mean(sub_block) for sub_block in sub_block_logprobs_list]
                            logprobs_softmax_list = self.softmax(logprobs_mean_list)
                            indices = list(range(len(logprobs_softmax_list)))
                            selected_idx = random.choices(indices, weights=logprobs_softmax_list, k=1)[0]
                            fallback_idx = 0
                            for i in range(selected_idx):
                                fallback_idx += len(sub_block_logprobs_list[i])
                            # print(f"[fallback_idx: {fallback_idx}")  
                        else:
                            raise NotImplementedError(f"fallback_policy {fallback_policy} not implemented")                        
                        
                        truncate_len = new_sample.output_len - fallback_idx
                        
                        new_sample.truncate_response(truncate_len=truncate_len)                        
                        new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            
                        new_sample.status = SampleStatus.TO_INFER
                        
                        if remaining_width_budget >= fixed_step_width:
                            for _ in range(fixed_step_width):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                                fallback_counter[root_idx] += 1
                            remaining_width_budget -= fixed_step_width
                        else:
                            for _ in range(remaining_width_budget):
                                samples_to_infer.append(copy.deepcopy(new_sample))
                                fallback_counter[root_idx] += 1
                            remaining_width_budget = 0
                
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
        print(f"time cost: {end_time - start_time} seconds")
        print(f"total number of trajectory: {total_n_trajectory}")
        print(f"average time per trajectory: {(end_time - start_time) / total_n_trajectory} seconds")
        response_token_lens = [s.actual_response_len for s in sum(finished_samples.values(), [])]
        total_tokens = sum(response_token_lens)
        print(f"tokens/sec: {total_tokens / (end_time - start_time)}")
        # assert sum(len(finished_samples[root_idx]) for root_idx in finished_samples) == n_target_trajectory, "finished_samples should be equal to n_target_trajectory"
        # 统计 finished_reason：
        finished_reason_counter = defaultdict(int)
        response_lens = []
        for root_idx in finished_samples:
            for finished_sample in finished_samples[root_idx]:
                finished_reason_counter[finished_sample.finished_reason] += 1
                response_lens.append(finished_sample.output_len)
        print(f"For {total_n_trajectory} responses, [finished reason]:\n{finished_reason_counter}")
        print(f"response length mean, min, max: {np.mean(response_lens)}, {np.min(response_lens)}, {np.max(response_lens)}")
        
        
        # 初始化 response 列表
        # 对齐原有输出格式
        response = []
        tree_indices = []
        response_root_indices = []

        # 收集样本统计信息
        sample_stats = []
        for root_idx in range(len(prompts)):
            for sample in finished_samples[root_idx]:
                sample_stats.append({
                    "tree_idx": sample.tree_idx,
                    "depth": sample.depth,
                    "token_count_per_step": sample.token_count_per_step,
                    "total_tokens": sum(sample.token_count_per_step),
                    "finished_reason": str(sample.finished_reason),
                    "output_len": sample.output_len
                })
                
                if sample.output_len > self.max_output_len:
                    print(f"sample {sample.tree_idx} output length {sample.output_len} > max_output_len {self.max_output_len}, truncate response")
                    print(f"[response before truncation]:\n{self.tokenizer.decode(sample.full_response_token_ids, skip_special_tokens=False)}")
                    truncate_len = sample.output_len - self.max_output_len
                    sample.truncate_response(truncate_len=truncate_len)
                response.append(sample.full_response_token_ids)
                response_root_indices.append(root_idx) # 保存每一条 response 对应的 root idx，因为后面要用来做 repeat（数量不一致的情况不能直接 repeat interleave。）
                tree_indices.append(copy.copy(sample.tree_idx))
                
        # 添加到 non_tensor_batch
        non_tensor_batch["sample_stats"] = np.array(sample_stats, dtype=object)

        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=self.max_output_len).to(idx.device)
        if max_width > 1:
            # idx = _repeat_interleave(idx, max_width)
            # attention_mask = _repeat_interleave(attention_mask, max_width)
            # position_ids = _repeat_interleave(position_ids, max_width)
            # batch_size = batch_size * max_width
            # for key in non_tensor_batch.keys():
            #     non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], max_width)
            
            # use _repeat_by_indices to support different number of responses per root node
            idx = _repeat_by_indices(idx, response_root_indices)
            attention_mask = _repeat_by_indices(attention_mask, response_root_indices)
            position_ids = _repeat_by_indices(position_ids, response_root_indices)
            batch_size = idx.size(0)
            for key in non_tensor_batch.keys():
                non_tensor_batch[key] = _repeat_by_indices(non_tensor_batch[key], response_root_indices)  
        
        # return the tree_idx for further use in the trainer
        non_tensor_batch["tree_idx"] = np.array(tree_indices, dtype=object)
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
        
        # import orjsonl
        # orjsonl.save(f"q-8_r-128.tree.efficiency.jsonl", step_efficiency_metrics)

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
        
        meta_info['n_finished_traj_at_first_fallback'] = np.array(list(n_finished_traj_at_first_fallback.values()), dtype=object)
        meta_info['fallback_counter'] =  np.array(list(fallback_counter.values()), dtype=object)
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    
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
        cal_prob_block_size=128,
        max_fallback_window=-1,
    ):
        '''
        Fallback to certan index and truncate the response, than change the sample.status
        '''
        if fallback_policy == "random":
            # 随机 fallback 到某个位置
            fallback_idx = np.random.randint(0, 1+max(0, (new_sample.output_len-1))//max_token_per_step) * max_token_per_step
        elif fallback_policy == "min_logprob_token":
            new_sample_response_logprobs = new_sample.logprobs
            new_sample_response_logprobs = [x[0] for x in new_sample_response_logprobs]
            fallback_idx = np.argmin(new_sample_response_logprobs)
        elif fallback_policy == "block":
            new_sample_response_logprobs = new_sample.logprobs
            new_sample_response_logprobs = [x[0] for x in new_sample_response_logprobs]
            sub_block_logprobs_list = [new_sample_response_logprobs[i:i+cal_prob_block_size] for i in range(0, len(new_sample_response_logprobs), cal_prob_block_size)]
            logprobs_mean_list = [np.mean(sub_block) for sub_block in sub_block_logprobs_list]
            logprobs_softmax_list = self.softmax(logprobs_mean_list)
            indices = list(range(len(logprobs_softmax_list)))
            selected_idx = random.choices(indices, weights=logprobs_softmax_list, k=1)[0]
            fallback_idx = 0
            for i in range(selected_idx):
                fallback_idx += len(sub_block_logprobs_list[i])
            # print(f"[fallback_idx: {fallback_idx}")  
        else:
            raise NotImplementedError(f"fallback_policy {fallback_policy} not implemented")
        
        # fallback_idx = np.random.randint(0, new_sample.output_len)
        truncate_len = new_sample.output_len - fallback_idx
        if max_fallback_window > 0:
            truncate_len = min(max_fallback_window*max_token_per_step, truncate_len)
            fallback_idx = new_sample.output_len - truncate_len
            
        new_sample.truncate_response(truncate_len=truncate_len)
        
        n_remaining_block = int(np.ceil(fallback_idx/max_token_per_step ))
        new_sample.truncate_tree_idx(new_sample.depth-n_remaining_block)
        
        new_sample.status = SampleStatus.TO_INFER 
        return new_sample   
        

    def generate_sequences_by_token_budget(
        self, 
        prompts: DataProto, 
        fixed_step_width: int = 2,
        random_first_div_max: int = -1,
        max_token_per_step: int = 512, 
        max_depth: int = 6,
        max_width: int = 8, # ~= rollout n trajectory
        fallback_selection_policy: str = "integrity_weighted", # "integrity_first", "integrity_weighted", "random_pick"
        fallback_policy: str = "random",
        divergence_policy: str = "fixed_avg",
        divergence_budget_control: str = "by_fixed_div", # or "by_infer_step_token_budget"
        logprob_div_temperature: float = 1.0,
        cal_prob_block_size: int = 128,
        max_response_token: int = -1, # limit of the sum of the response token per prompt
        total_budget_policy: str = "by_response_token", # 
        **kwargs) -> DataProto:
        
        """
        implement tree inference.
        """
        
        assert total_budget_policy != "by_response_traj", f"not supprt {total_budget_policy=} in this function, buggy"
        if total_budget_policy == "by_response_token":
            print(f"[WARNING]: `fixed_step_width` refers to the fixed number of paths for each prompt for one inference step with {total_budget_policy=}")
        
        # assert max_depth * max_token_per_step <= self.max_output_len, f"only supports max_depth * max_token_per_step (current: {max_depth * max_token_per_step}) <= max_output_len (current: {self.max_output_len}) ATM"
        
        calculate_entropy = kwargs.get("calculate_entropy", False)
        if calculate_entropy:
            assert kwargs.get("logprobs", -1) >= 0, "n_return_logprobs should be greater than 0 if calculate_entropy is true"
        
        start_time = time.time()
        
        # TODO: implement the tree search by max token
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info["eos_token_id"]
        eos_token_id = self.tokenizer.eos_token_id
        
        batch_size = idx.size(0)
        
        meta_info = {} # store the meta_info to return
        
        non_tensor_batch = prompts.non_tensor_batch
        vllm_inputs = _get_vllm_inputs(prompts, self.tokenizer.pad_token_id)
 
        # width_counter = {root_idx: 0 for root_idx in range(len(vllm_inputs))} # record number of finished samples for each root node
        finished_samples = {root_idx: [] for root_idx in range(batch_size)} # store the tuples of (finished response ids, reasoning tree batch idx, finished_status)
        
        fallback_counter = defaultdict(int) # 记录 fallback 的次数
        n_finished_traj_at_first_fallback = {}
        
        # ===================== 构造第一批输入 =====================
        samples_to_infer = []
        for root_idx in range(len(vllm_inputs)):
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
            
        if divergence_budget_control == "by_infer_step_token_budget":
            print(f"[WARNING] setting {divergence_budget_control=}, now use the {fixed_step_width=} as the total maximum inference token budget, and {max_token_per_step=} as the budget for vLLM params for each input.")
            first_step_div = fixed_step_width//max_token_per_step//len(samples_to_infer)
            if first_step_div <= 1:
                print(f"[WARNING] average first divergence {first_step_div=} <=1, please set a larger value of {fixed_step_width=}.")
            first_step_div = max(1, first_step_div) # at least one
            first_step_div = min(first_step_div, div_upper_bound)
            samples_to_infer = _repeat_list_interleave(samples_to_infer, first_step_div)
        elif random_first_div_max > fixed_step_width:
            assert random_first_div_max <= div_upper_bound, f"{random_first_div_max=} should be less than or equal to {div_upper_bound=}"
            duplicated_indices = []
            for sample in samples_to_infer:
                # 这里的 sample.tree_idx 是 str 类型
                first_step_div = np.random.randint(fixed_step_width, random_first_div_max+1) 
                duplicated_indices.extend([sample.root_node]*first_step_div)
            # 这里的 duplicated_indices 是 list[int]，每个元素是 sample.root_node 的索引
            samples_to_infer = _repeat_list_by_indices(samples_to_infer, duplicated_indices)
            print(f"first step rollout, duplicate {len(vllm_inputs)} samples to infer: {len(samples_to_infer)}")
        else:
            samples_to_infer = _repeat_list_interleave(samples_to_infer, fixed_step_width)
                
        next_infer_step = 1
        samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step=next_infer_step)
          
        step_start_times = [start_time]  # 记录每个 step 开始的时间
        step_efficiency_metrics = []
        
        # ===================== 开始推理，直到队列中为空 =====================
        while len(samples_to_infer) > 0:
            assert kwargs.get("n", 1) == 1, "n must be 1 for tree search, use repeat_interleave to generate multiple samples"
            kwargs["max_tokens"] = max_token_per_step
            
            # if isinstance(vllm_inputs[0], list):
            #     vllm_inputs = [{"prompt_token_ids": vllm_input} for vllm_input in vllm_inputs]
            vllm_inputs = get_vllm_inputs_from_samples(samples_to_infer)
            print(f"vllm_inputs to infer: {len(vllm_inputs)}")
            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                current_step_end_time = time.time()
                step_duration = current_step_end_time - step_start_times[-1]
                step_start_times.append(current_step_end_time)
                step_efficiency_metrics.append({
                    "step_duration": step_duration,
                    "max_token_per_step": kwargs["max_tokens"],
                    "n_inputs": len(vllm_inputs),
                    "average_input_tokens": float(np.mean([len(list(sample.input_ids)) for sample in samples_to_infer])),
                    "n_outputs": len(outputs),
                    "average_output_tokens": float(np.mean([len(list(output.outputs[0].token_ids)) for output in outputs])),
                })
            samples_last_step = copy.deepcopy(samples_to_infer)
            samples_to_infer = []

            samples_to_go_deeper = {root_idx: [] for root_idx in range(batch_size)} # 记录下一个 infer step 中每个 root node 需要深入的路径。       
            finished_samples_this_step_by_root_node = defaultdict(int) # 记录当前 step 中每个 root node 的 finished 数量，用于做分岔预算转移
            
            # ===================== 逐个处理推理结果 =====================
            for infer_batch_idx, (sample, output) in enumerate(zip(samples_last_step, outputs)):
                assert len(output.outputs) == 1, "vllm should only generate one output"
                # 取出当前 infer step 结果
                response_id = list(output.outputs[0].token_ids)
                response_str = output.outputs[0].text
                
                if calculate_entropy:
                    response_logprobs = _get_generated_logprobs_from_vllm_output(output.outputs[0])
                else:
                    response_logprobs = [[LOGPROG_PLACEHOLDER] for _ in range(max(1, kwargs.get("logprobs", 0)))] * len(response_id)
                
                sample = copy.deepcopy(sample) # 必须要 copy，否则会影响后续的判断
                sample.extend_response(response_id, response_logprobs)
                    
                last_eos_idx = _find_subtokens_idx_reverse(response_id, [self.tokenizer.eos_token_id])
                boxed_answer = extract_last_boxed(response_str)
                acutal_response_len = sample.actual_response_len # actual len 包含了之前 infer step 得到的回复
                
                # 如果 response 中存在完整 \\boxed{}
                if boxed_answer is not None:
                    sample.finished_reason = FinishedReason.FINISHED
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                # 如果 response 中存在 EOS
                elif last_eos_idx >= 0:
                    sample.finished_reason = FinishedReason.EARLY_STOPPED_BY_EOS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                # 如果没有 answer，则认为这条路径没有结束，判断是否要继续 rollout
                # 如果实际 response 长度 >= max_output_len，则认为这条路径已经结束
                elif acutal_response_len >= self.max_output_len:
                    print(f"{sample.tree_idx}: actual response length >= max_output_len, stop")
                    if acutal_response_len > self.max_output_len:
                        truncate_len = acutal_response_len - self.max_output_len
                        sample.truncate_response(truncate_len=truncate_len)     
                    sample.finished_reason = FinishedReason.MAX_OUTPUT_TOKENS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                elif sample.depth >= max_depth:
                    print(f"{sample.tree_idx}: reach max_depth, stop")
                    sample.finished_reason = FinishedReason.MAX_INFER_STEPS
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                res_has_repretition, repeated_substr = _has_repetition(response_str)
                if res_has_repretition:
                    # print(f"{sample.tree_idx}: response contains repetition, stop, [response]:\n{response_str}")
                    # print(repeated_substr)
                    sample.finished_reason = FinishedReason.REPETITION_STOP
                    sample.status = SampleStatus.FINISHED
                    finished_samples[sample.root_node].append(sample)
                    finished_samples_this_step_by_root_node[sample.root_node] += 1
                    continue
                else:
                    # 如果还有足够的 token budget，则继续 rollout
                    assert sample.status == SampleStatus.TO_INFER, f"sample {sample.tree_idx} should be TO_INFER, but is {sample.status}"
                    samples_to_go_deeper[sample.root_node].append(copy.deepcopy(sample))
            
            # counting for next infer step, default 0
            root_to_infer_count = defaultdict(int)
            root_to_finished_tokens = defaultdict(int)
            for root_node in finished_samples:
                for finished_sample in finished_samples[root_node]:
                    root_to_finished_tokens[root_node] += finished_sample.actual_response_len

            # ===================== 遍历所有 root_node，查看是否有需要分岔或回退的路线 ===================== 
            
            for root_node in finished_samples:
                # `samples_to_go_deeper` 代表分岔之前的 active path，不加入下一次 infer 的 sample
                remaining_tokens = max_response_token - root_to_finished_tokens[root_node]
                if len(finished_samples[root_node]) > 0:
                    avg_finished_len = root_to_finished_tokens[root_node]//len(finished_samples[root_node]) # 用已完成的 path len 来预估后面的预算
                else:
                    avg_finished_len = 0
                # ===================== 至少还有一条平均路径预算的情况下，如果没有现存路径，做回溯捞路径 (默认按优先级回溯) ===================== 
                n_path_from_fallback = 0
                if remaining_tokens >= avg_finished_len and len(samples_to_go_deeper[root_node]) <=0 and fallback_policy != "no_fallback":
                    # 记录第一次发生 fallback 的时候有多少条正常跑完的 trajectory
                    if root_node not in n_finished_traj_at_first_fallback:
                        n_finished_traj_at_first_fallback[root_node] = len(finished_samples[root_node])

                    # ===================== 根据策略从已经结束的路径中选择进行回溯 ===================== 
                    # 支持三种模式： "integrity_ordered", "integrity_weighted", "random_pick"
                    # "integrity_ordered" 按优先级逐条加回，直到用完预算或者没有路径：boxed, eos, max_len, early stopped
                    if fallback_selection_policy == "integrity_ordered":
                        finished_samples[root_node] = sorted(finished_samples[root_node], key=lambda s: s.finished_reason)
                    # "integrity_weighted" 按优先级对已完成 path 进行采样，直到用完预算或者没有路径
                    elif fallback_selection_policy == "integrity_weighted":
                        finished_samples[root_node] = _gumbel_topk_permutation(finished_samples[root_node])
                    # "random_pick"
                    elif fallback_selection_policy == "random_pick":
                        random.shuffle(finished_samples[root_node])
                    else:
                        raise NotImplementedError
                                            
                    # 根据指定的顺序，将 fallback 路径加入到 infer 队列
                    estimated_remaining_tokens = remaining_tokens # 用于计算还是否有预算做回溯
                    for finished_sample in finished_samples[root_node]:
                        if estimated_remaining_tokens <= 0:
                            break
                        new_sample = copy.deepcopy(finished_sample)
                        new_sample = self.create_fallback_sample_by_policy(new_sample, max_token_per_step, fallback_policy, cal_prob_block_size)
                        samples_to_go_deeper[root_node].append(new_sample)
                        if divergence_budget_control == "by_fixed_div":    
                            estimated_remaining_tokens -= avg_finished_len * fixed_step_width
                        else:
                            raise NotImplementedError
                    n_path_from_fallback = len(samples_to_go_deeper[root_node])
                    print(f"{root_node=} has {remaining_tokens=}, but no go-deeper path. Add {n_path_from_fallback} fallback samples to infer.")
                
                if len(samples_to_go_deeper[root_node]) <=0:
                    continue
                # ===================== 如果有需要推理的路径，针对每个 root 计算 infer step divergence ===================== 
                # 根据 ongoing sample 的平均长度，和 infererence token，计算最多能有多少分岔。
                # 计算 `total_divergence_this_step`，也就是本次 infer step 的分岔预算。
                if divergence_budget_control == "by_infer_step_token_budget":
                    raise NotImplementedError(f"not support by_response_token + by_infer_step_token_budget")
                    # # the budget is pre-defined by total number of output tokens in `fixed_step_width`, the upper bound is definied by the final trajectory numbers.
                    # token_budget_per_query = fixed_step_width//len(samples_to_go_deeper.keys()) # 这个是 cross-query 的？
                    # remaining_width_budget = div_upper_bound - len(finished_samples[root_node]) - root_to_infer_count.get(root_node, 0)
                    
                    # total_divergence_this_step = min(remaining_width_budget, token_budget_per_query//max_token_per_step) # 如果超过了 max_width，则不再继续 rollout
                    # total_divergence_this_step = max(total_divergence_this_step, len(samples_to_go_deeper[root_node])) # 每个路径至少要能继续往下分岔一个
                elif divergence_budget_control == "by_fixed_div":
                    # 如果 budget 已经用完，则完成已有 sample 就结束
                    if remaining_tokens <= 0:
                        total_divergence_this_step = len(samples_to_go_deeper[root_node])
                    else:
                        # the budget is transferred from the finished trajectories, the upper bound is definied by the final trajectory numbers.                        
                        estimated_tokens_for_one_div = sum([todo_sample.actual_response_len + 0.8*max_token_per_step for todo_sample in samples_to_go_deeper[root_node]])//len(samples_to_go_deeper[root_node])
                        remaining_width_budget = remaining_tokens // estimated_tokens_for_one_div # 用预估的新长度计算分岔上限。有可能是 0，此时要把剩余路径都 roll 完
                        if remaining_width_budget <= len(samples_to_go_deeper[root_node]):
                            total_divergence_this_step = len(samples_to_go_deeper[root_node])
                        elif n_path_from_fallback > 0:
                            total_divergence_this_step = min([remaining_width_budget, fixed_step_width * len(samples_to_go_deeper[root_node])])
                        else:
                            # 将上次 infer 已经完成的 path 的预算转移给其他 active path。
                            extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * fixed_step_width
                            total_divergence_this_step = min([remaining_width_budget, extra_allow_divergence + fixed_step_width * len(samples_to_go_deeper[root_node])])
                            
                        MININUM_DIV = 4 # 这个对速度好像没有什么帮助
                        total_divergence_this_step = max(MININUM_DIV, total_divergence_this_step)
                        
                        print(f"root {root_node}; already finished {len(finished_samples[root_node])}; number of samples finishing in this step: {finished_samples_this_step_by_root_node[root_node]}; {remaining_tokens=}; {remaining_width_budget=}; path to go deeper: {len(samples_to_go_deeper[root_node])}; divergence for next round {total_divergence_this_step=}")
                else:
                    raise NotImplementedError(f"{divergence_budget_control=} not implemented")
                
                if divergence_policy == "fixed_avg":
                    average_divergence = total_divergence_this_step // len(samples_to_go_deeper[root_node])
                    remainder_divergence = total_divergence_this_step % len(samples_to_go_deeper[root_node])
                    # 将 total_divergence_this_step 的预算，均匀分配给 samples_to_go_deeper[root_node] 中的每个样本
                    for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                        if path_idx + 1 == len(samples_to_go_deeper[root_node]):
                            n_rollout = average_divergence + remainder_divergence
                        else:
                            n_rollout = average_divergence
                        n_rollout = int(max(1, n_rollout)) # 至少给一个预算
                        for _ in range(n_rollout):
                            new_sample = copy.deepcopy(sample)
                            new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            samples_to_infer.append(new_sample)
                            root_to_infer_count[root_node] += 1
                elif divergence_policy == "logprob_weighted_div":
                    log_probs = []                # (N, T_i) 列表或张量
                    for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                        log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                    cum_logprob = torch.tensor([lp.sum() for lp in log_probs])  # shape (N,)
                    # 先做一个 shift，防止极端负值下溢
                    shifted = -1.0 * (cum_logprob - cum_logprob.min())
                    weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                    divergence_budgets = weight_to_discrete_allocate(weights, total_divergence_this_step)
                    for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                        for _ in range(div_budget):
                            new_sample = copy.deepcopy(sample)
                            new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            samples_to_infer.append(new_sample)
                            root_to_infer_count[root_node] += 1
                elif divergence_policy == "inverse_logprob_weighted_div":
                    # 如果 logprob 越大，分岔权重越大
                    log_probs = []                # (N, T_i) 列表或张量
                    for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                        log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                    cum_logprob = torch.tensor([lp.sum() for lp in log_probs])  # shape (N,)
                    # 先做一个 shift，防止极端负值下溢
                    shifted = 1.0 * (cum_logprob - cum_logprob.min())
                    weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                    divergence_budgets = weight_to_discrete_allocate(weights, total_divergence_this_step)
                    for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                        for _ in range(div_budget):
                            new_sample = copy.deepcopy(sample)
                            new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            samples_to_infer.append(new_sample)
                            root_to_infer_count[root_node] += 1
                elif divergence_policy == "norm_logprob_weighted_div":
                    log_probs = []                # (N, T_i) 列表或张量
                    for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                        log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                    cum_logprob = torch.tensor([lp.mean() for lp in log_probs])  # shape (N,)
                    # 先做一个 shift，防止极端负值下溢
                    shifted = -1.0 * (cum_logprob - cum_logprob.min())
                    weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                    divergence_budgets = weight_to_discrete_allocate(weights, total_divergence_this_step)
                    for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                        for _ in range(div_budget):
                            new_sample = copy.deepcopy(sample)
                            new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            samples_to_infer.append(new_sample)
                            root_to_infer_count[root_node] += 1
                elif divergence_policy == "norm_inv_logprob_weighted_div":
                    remaining_width_budget = max_width - len(finished_samples[root_node]) - root_to_infer_count.get(root_node, 0)
                    extra_allow_divergence = finished_samples_this_step_by_root_node[root_node] * fixed_step_width                    
                    total_divergence_this_step = min([remaining_width_budget, extra_allow_divergence + fixed_step_width * len(samples_to_go_deeper[root_node])])

                    log_probs = []                # (N, T_i) 列表或张量
                    for path_idx, sample in enumerate(samples_to_go_deeper[root_node]):
                        log_probs.append(torch.tensor([x[0] for x in sample.logprobs]))
                    cum_logprob = torch.tensor([lp.mean() for lp in log_probs])  # shape (N,)
                    # 先做一个 shift，防止极端负值下溢
                    shifted = 1.0 * (cum_logprob - cum_logprob.min())
                    weights = torch.softmax(shifted/logprob_div_temperature, dim=0)   # shape (N,)
                    divergence_budgets = weight_to_discrete_allocate(weights, total_divergence_this_step)
                    for path_idx, (sample, div_budget) in enumerate(zip(samples_to_go_deeper[root_node], divergence_budgets)):
                        for _ in range(div_budget):
                            new_sample = copy.deepcopy(sample)
                            new_sample.input_ids = new_sample.init_input_ids + new_sample.full_response_token_ids
                            samples_to_infer.append(new_sample)
                            root_to_infer_count[root_node] += 1
                else:
                    raise NotImplementedError(f"divergence_policy {divergence_policy} not implemented")
                                
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
        print(f"time cost: {end_time - start_time} seconds")
        print(f"total number of trajectory: {total_n_trajectory}")
        print(f"average time per trajectory: {(end_time - start_time) / total_n_trajectory} seconds")
        response_token_lens = [s.actual_response_len for s in sum(finished_samples.values(), [])]
        total_tokens = sum(response_token_lens)
        print(f"tokens/sec: {total_tokens / (end_time - start_time)}")
        # assert sum(len(finished_samples[root_idx]) for root_idx in finished_samples) == n_target_trajectory, "finished_samples should be equal to n_target_trajectory"
        # 统计 finished_reason：
        finished_reason_counter = defaultdict(int)
        response_lens = []
        for root_idx in finished_samples:
            print(f"{root_idx=} finished samples {len(finished_samples[root_idx])}")
            for finished_sample in finished_samples[root_idx]:
                finished_reason_counter[finished_sample.finished_reason] += 1
                response_lens.append(finished_sample.output_len)
        print(f"For {total_n_trajectory} responses, [finished reason]:\n{finished_reason_counter}")
        print(f"response length mean, min, max: {np.mean(response_lens)}, {np.min(response_lens)}, {np.max(response_lens)}")
        
        
        # 初始化 response 列表
        # 对齐原有输出格式
        response = []
        tree_indices = []
        response_root_indices = []

        # 收集样本统计信息
        sample_stats = []
        for root_idx in range(len(prompts)):
            for sample in finished_samples[root_idx]:
                sample_stats.append({
                    "tree_idx": sample.tree_idx,
                    "depth": sample.depth,
                    "token_count_per_step": sample.token_count_per_step,
                    "total_tokens": sum(sample.token_count_per_step),
                    "finished_reason": str(sample.finished_reason),
                    "output_len": sample.output_len
                })
                
                if sample.output_len > self.max_output_len:
                    print(f"sample {sample.tree_idx} output length {sample.output_len} > max_output_len {self.max_output_len}, truncate response")
                    print(f"[response before truncation]:\n{self.tokenizer.decode(sample.full_response_token_ids, skip_special_tokens=False)}")
                    truncate_len = sample.output_len - self.max_output_len
                    sample.truncate_response(truncate_len=truncate_len)
                response.append(sample.full_response_token_ids)
                response_root_indices.append(root_idx) # 保存每一条 response 对应的 root idx，因为后面要用来做 repeat（数量不一致的情况不能直接 repeat interleave。）
                tree_indices.append(copy.copy(sample.tree_idx))
                
        # 添加到 non_tensor_batch
        non_tensor_batch["sample_stats"] = np.array(sample_stats, dtype=object)

        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=self.max_output_len).to(idx.device)
        if max_width > 1:
            # idx = _repeat_interleave(idx, max_width)
            # attention_mask = _repeat_interleave(attention_mask, max_width)
            # position_ids = _repeat_interleave(position_ids, max_width)
            # batch_size = batch_size * max_width
            # for key in non_tensor_batch.keys():
            #     non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], max_width)
            
            # use _repeat_by_indices to support different number of responses per root node
            idx = _repeat_by_indices(idx, response_root_indices)
            attention_mask = _repeat_by_indices(attention_mask, response_root_indices)
            position_ids = _repeat_by_indices(position_ids, response_root_indices)
            batch_size = idx.size(0)
            for key in non_tensor_batch.keys():
                non_tensor_batch[key] = _repeat_by_indices(non_tensor_batch[key], response_root_indices)  
        
        # return the tree_idx for further use in the trainer
        non_tensor_batch["tree_idx"] = np.array(tree_indices, dtype=object)
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
        
        # import orjsonl
        # orjsonl.save(f"q-8_r-128.tree.efficiency.jsonl", step_efficiency_metrics)

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
        
        meta_info['n_finished_traj_at_first_fallback'] = np.array(list(n_finished_traj_at_first_fallback.values()), dtype=object)
        meta_info['fallback_counter'] =  np.array(list(fallback_counter.values()), dtype=object)
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

def calculate_response_token_len(output_batch: DataProto, pad_token_id: int):
    # for each tensor in output_batch.batch["responses"], calculate the length of the tensor
    # skip calculating the length of the padding tokens
    response_token_len = []
    for i in range(len(output_batch.batch["responses"])):
        response_token_len.append(torch.where(output_batch.batch["responses"][i] != pad_token_id)[0].shape[0])
    return response_token_len

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import json

# unit test 
if __name__ == "__main__":
    """
    Example usage:

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python recipe/treepo/vllm_rollout_tree.py \
        --model_path /path/to/models/Qwen2.5-Math-7B \
        --prompt_path /path/to/prompts.parquet
    ```

    For sequential generation:

    ```bash
    CUDA_VISIBLE_DEVICES=7 python recipe/treepo/vllm_rollout_tree.py \
        --model_path /path/to/models/Qwen2.5-Math-7B \
        --infer_mode sequential \
        --n_prompts 8 \
        --calculate_entropy
    ```
    """
    '''
    add these arguments for cmd
    def generate_sequences_tree_deepth_first_vanilla_mode(
        self, 
        prompts: DataProto, 
        fixed_step_width: int = 2,
        random_first_div_max: int = -1,
        max_token_per_step: int = 512, 
        max_depth: int = 6,
        max_width: int = 8, # ~= rollout n trajectory
        force_answer_remaining_token: int = 128,
        # force_answer_threshold: int = 2,
        fallback_policy: str = "random",
        divergence_policy: str = "fixed_avg",
        divergence_budget_control: str = "by_final_trajectory", # or "by_infer_step_token_budget"
        logprob_div_temperature: float = 1.0,
        cal_prob_block_size: int = 128,
        **kwargs) -> DataProto:
    '''

    # import hydra
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ["USE_VLLM_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, required=True, help="Model path compatible with vLLM.")
    args.add_argument("--model_type", type=str, choices=["step_model", "base_model"], default="base_model", help="model type, choose from 'step_model' or 'base_model'")
    # args.add_argument("--config_path", type=str, default="", help="ymal config file,use hydra to load")
    args.add_argument("--infer_mode", type=str, default="tree_by_max_token", help="infer mode, choose from 'sequential' or 'tree_by_max_token'")
    args.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor parallel size")
    args.add_argument("--vllm_gpu_util", type=float, default=0.6, help="")
    args.add_argument("--n_prompts", type=int, default=16, help="number of prompts to generate")
    args.add_argument("--n_rollout", type=int, default=16, help="number of prompts to generate")
    args.add_argument("--calculate_entropy", action="store_true", help="calculate entropy")
    args.add_argument("--n_return_logprobs", type=int, default=0, help="number of logprobs to return")
    args.add_argument("--tree_default_divergence", type=int, default=16, help="number of divergence for tree generation")
    args.add_argument("--tree_max_depth", type=int, default=6, help="max depth for tree generation (inference step)")
    args.add_argument("--tree_infer_max_token", type=int, default=512, help="max token for each step of tree generation inference")
    args.add_argument("--tree_force_answer_remaining_token", type=int, default=256, help="force answer remaining token for tree generation")
    args.add_argument("--tree_divergence_policy", type=str, default="fixed_avg")
    args.add_argument("--tree_divergence_budget_control", type=str, default="by_fixed_div")
    args.add_argument("--tree_total_budget_control", type=str, default="by_response_traj")
    args.add_argument("--tree_max_response_token", type=int, default=-1, help="maximum generated tokens")
    args.add_argument("--seed", type=int, default=8, help="seed")
    args.add_argument("--test_output_file", type=str, default=None, help="test output file")
    args.add_argument("--log_prob_save_path", type=str, default=None, help="a path to save the log probabilities of the generated sequences in numpy")
    args.add_argument("--prompt_path", type=str, default=None, help="Optional parquet file containing prompts in TreePO format.")
    args = args.parse_args()

    seed_everything(args.seed)
    
    n_prompts = args.n_prompts
    if args.prompt_path:
        df = pd.read_parquet(args.prompt_path)
        prompts = [
            df.iloc[i]["prompt"][1]["content"].replace("Question: ", "")
            for i in range(min(n_prompts, len(df)))
        ]
        if len(prompts) < n_prompts:
            raise ValueError(
                f"Requested {n_prompts} prompts but only found {len(prompts)} in {args.prompt_path}."
            )
    else:
        prompts = [
            "The operation $\\otimes$ is defined for all nonzero numbers by $a \\otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)].$",
            "Doug constructs a square window using eight equal-size panes of glass. Each pane has height-to-width ratio $5:2$, with two-inch borders between panes. What is the side length of the window?",
            "Let $P(x)$ be a polynomial of degree $3n$ such that $P(3k) = 2$, $P(3k+1) = 1$, and $P(3k+2) = 0$ for integers $k$ with $0 \le k \le n-1$, and $P(3n+1) = 730$. Determine $n$.",
            "Let $f(x)=ax^2-\\sqrt{2}$ with $a>0$ and $f(f(\\sqrt{2})) = -\\sqrt{2}$. Find $a$.",
        ]
        if n_prompts > len(prompts):
            raise ValueError(
                "Provide --prompt_path to supply enough prompts for the requested --n_prompts value."
            )
        prompts = prompts[:n_prompts]
    
    
    # generation_mode = "sequential"
    # generation_mode = "tree_by_max_token"
    generation_mode = args.infer_mode
    
    # load config
    vllm_test = vLLMTest(args.model_path, tensor_parallel_size=args.tensor_parallel_size, model_type=args.model_type, gpu_util=args.vllm_gpu_util)
    input_batch = vllm_test.prepare_input(prompts)
    start_time = time.time()
    if generation_mode == "sequential":
        rollout_sampling_params = {
            "n": args.n_rollout,
        }
        if args.calculate_entropy:
            rollout_sampling_params["calculate_entropy"] = True
            rollout_sampling_params["logprobs"] = len(vllm_test.tokenizer) if args.n_return_logprobs < 0 else args.n_return_logprobs
        output_batch = vllm_test.generate_sequence_sequential(input_batch, **rollout_sampling_params)
    else:
        rollout_sampling_params = {
            "fixed_step_width": args.tree_default_divergence,
            "divergence_policy": args.tree_divergence_policy,
            "divergence_budget_control": args.tree_divergence_budget_control,
            "max_token_per_step": args.tree_infer_max_token,
            "max_depth": args.tree_max_depth,
            "max_width": args.n_rollout,
            "force_answer_remaining_token": args.tree_force_answer_remaining_token,
            "total_budget_policy": args.tree_total_budget_control,
            "max_response_token": args.tree_max_response_token,
        }
        if args.calculate_entropy:
            rollout_sampling_params["calculate_entropy"] = True
            rollout_sampling_params["logprobs"] = len(vllm_test.tokenizer) if args.n_return_logprobs < 0 else args.n_return_logprobs
        
        if args.model_type == "base_model":
            if args.tree_total_budget_control == "by_response_token":
                output_batch = vllm_test.generate_sequences_by_token_budget(input_batch, **rollout_sampling_params)
            else:
                output_batch = vllm_test.generate_sequences_tree_deepth_first_vanilla_mode(input_batch, **rollout_sampling_params)
        elif args.model_type == "step_model": 
            output_batch = vllm_test.generate_sequences_tree_deepth_first(input_batch, **rollout_sampling_params)
    end_time = time.time()
    
    print("--------------------------------")
    print("example output:")
    # decode the output
    for i in range(len(output_batch.batch["responses"])):
        print(vllm_test.tokenizer.decode(output_batch.batch["responses"][i], skip_special_tokens=True))
        break
    print("--------------------------------")
    print(f"Total {len(output_batch)} samples generated for {len(prompts)} prompts")
    if args.calculate_entropy and args.n_return_logprobs > 0:
        print("-----------Entropy Stats-------------")
        print(f"Average entropy for each prompt:")
        all_entropy = []
        for i in range(0, len(output_batch), args.n_rollout):
            prompt_level_entropy = []
            prompt_level_entropy_with_placeholder = []
            for j in range(args.n_rollout): 
                topk_entropy = compute_sequence_entropy_from_topk(output_batch.non_tensor_batch['response_logprobs'][i+j])
                prompt_level_entropy.append(topk_entropy)
                topk_entropy_with_placeholder = compute_sequence_entropy_from_topk(output_batch.non_tensor_batch['response_logprobs'][i+j], skip_placeholder=False)
                prompt_level_entropy_with_placeholder.append(topk_entropy_with_placeholder)
            print(f"Average entropy for prompt {i//args.n_rollout}: {np.mean(prompt_level_entropy)}")
            all_entropy.append(np.mean(prompt_level_entropy))
        print(f"Average entropy for ALL prompts: {np.mean(all_entropy)}")
        print("--------------------------------")
    print(f"Time taken for {generation_mode} generation to get {len(output_batch)} samples: {end_time - start_time} seconds")
    
    response_token_len = calculate_response_token_len(output_batch, vllm_test.tokenizer.pad_token_id)
    print(f"Average response token length: {np.mean(response_token_len)}")
    print(f"Average time per token: {(end_time - start_time) / np.sum(response_token_len)} seconds")
    
    # 添加统计信息
    print("\n=== Tree Search Statistics ===")
    # 从 output_batch 获取 non_tensor_batch
    non_tensor_batch = output_batch.non_tensor_batch
    
    # 从 sample_stats 中获取统计信息
    if generation_mode == "tree_by_max_token":
        depths = [stat["depth"] for stat in non_tensor_batch["sample_stats"]]
        token_counts_by_step = defaultdict(list)
        for stat in non_tensor_batch["sample_stats"]:
            for step_idx, token_count in enumerate(stat["token_count_per_step"]):
                token_counts_by_step[step_idx].append(token_count)
    
        # 打印统计信息
        print("\nSearch Depth Distribution:")
        depth_counts = np.bincount(depths)
        depth_counts_dict = {}
        for depth in range(len(depth_counts)):
            count = depth_counts[depth]
            if count > 0:
                print(f"Depth {depth}: {count} samples ({count/len(depths)*100:.1f}%)")
                depth_counts_dict[str(depth)] = int(count)
        
        print("\nToken Count Distribution per Step:")
        for step_idx, token_counts in sorted(token_counts_by_step.items()):
            print(f"Step {step_idx}:")
            print(f"  Mean: {np.mean(token_counts):.1f}")
            print(f"  Min: {np.min(token_counts)}")
            print(f"  Max: {np.max(token_counts)}")
            print(f"  Std: {np.std(token_counts):.1f}")
    
    if args.test_output_file:
        assert args.test_output_file.endswith(".json"), "test_output_file must end with .json"
        os.makedirs(os.path.dirname(args.test_output_file), exist_ok=True)
        
        # 首先将 args 中的参数写入到 json 文件中
        all_inputs = output_batch.batch.pop("prompts")
        all_responses = output_batch.batch.pop("responses")
        test_output = {
            "args": vars(args),
            "average_response_token_len": np.mean(response_token_len),
            "average_time_per_token": (end_time - start_time) / np.sum(response_token_len),
            "average_time_per_sequence": (end_time - start_time) / len(output_batch),
            "response_token_len": response_token_len,
            "time_cost": end_time - start_time,
            "search_depth_distribution": depth_counts_dict if generation_mode == "tree_by_max_token" else None,
            "token_count_distribution": {
                step_idx: {
                    "mean": float(np.mean(token_counts)),
                    "min": int(np.min(token_counts)),
                    "max": int(np.max(token_counts)),
                    "std": float(np.std(token_counts)),
                    "counts": token_counts
                }
                for step_idx, token_counts in token_counts_by_step.items()
            } if generation_mode == "tree_by_max_token" else None,
            "input_output_pairs": [
                {
                    "input": vllm_test.tokenizer.decode(all_inputs[i], skip_special_tokens=True),
                    "output": vllm_test.tokenizer.decode(all_responses[i], skip_special_tokens=True),
                }
                for i in range(len(output_batch))
            ]
        }
        if args.calculate_entropy:
            test_output["prompt_level_entropy"] = prompt_level_entropy
            test_output["average_entropy"] = np.mean(prompt_level_entropy)
            test_output["average_entropy_with_placeholder"] = np.mean(prompt_level_entropy_with_placeholder)
        with open(args.test_output_file, "w") as f:
            json.dump(test_output, f, indent=4, ensure_ascii=False)

    if args.log_prob_save_path:
        # 保存 log probabilities
        os.makedirs(os.path.dirname(args.log_prob_save_path), exist_ok=True)
        all_log_probs = output_batch.non_tensor_batch.get("response_logprobs", None)
        assert all_log_probs is not None, "Log probabilities are not available in the output batch. Please ensure 'calculate_entropy' is set to True during generation."
        all_log_probs = [np.array(all_log_probs[i]).flatten() for i in range(len(all_log_probs))]
        logprobs_by_query = {}  
        for start_idx in range(0, len(all_log_probs), args.n_rollout):
            query_idx = start_idx // args.n_rollout
            logprobs_by_query[query_idx] = all_log_probs[start_idx:start_idx+args.n_rollout]
            print(f"Query {query_idx} logprobs from {start_idx} to {start_idx + args.n_rollout - 1}")
        np.savez(args.log_prob_save_path, logprobs_by_query=logprobs_by_query)
        print(f"Log probabilities saved to {args.log_prob_save_path}")
        
        '''
        Example command for logprob distribution analysis:
        CUDA_VISIBLE_DEVICES=0,1,2,3 python recipe/treepo/vllm_rollout_tree.py \
            --model_type base_model \
            --model_path /path/to/models/Qwen2.5-Math-7B \
            --infer_mode sequential \
            --n_prompts 32 \
            --n_rollout 32 \
            --calculate_entropy \
            --n_return_logprobs 0 \
            --tensor_parallel_size 2 \
            --log_prob_save_path logs/vllm_rollout_base_model_logprob_analysis/sequential/n-prompts-32_rollout-32_log_probs.npz
        '''