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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score

import time
import ray
from typing import List, Dict
from ray.exceptions import GetTimeoutError  # 用于处理超时
from verl.workers.reward_manager.utils import reward_func_timeout_ray
import random

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        self.multi_proc = True
        self.timeout_seconds = 5
        self.default_negative_score = 0.0
        
    # 修改原始方法，使用 Ray
    def math_compute_score_parallel_with_ray(
        self, 
        data_sources, solution_strs, ground_truths, extra_infos,
        # first_rollout_rewards, second_reward_rates
    ):
        scores: List[float] = [self.default_negative_score] * len(solution_strs)
        extra_info_dict: Dict[str, List[float]] = {}  # Key -> list of values for the batch
        print(f"Scoring process started over {len(solution_strs)} samples, waiting for results...")

        futures = []
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]
            # first_rollout_reward = first_rollout_rewards[i]
            # second_reward_rate = second_reward_rates[i]
            if extra_info is None:
                extra_info = {}
            # extra_info['first_rollout_reward'] = first_rollout_reward
            # extra_info['second_reward_rate'] = second_reward_rate

            # 提交任务给 Ray
            future = reward_func_timeout_ray.remote(
                self.compute_score, self.timeout_seconds, data_source, solution_str, ground_truth, extra_info, 'strict', self.default_negative_score
            )
            futures.append(future)

        # default_fail_score = {"score": 0.0, "extra_info": {"is_filter": 1}}  # Default on error which should be filtered
        default_fail_score = {
            "score": self.default_negative_score, "acc": 0.0, 
            "extra_info": {
                "pred": '[default verify failed placeholder]', 
                "is_boxed_ratio": 0.0
            }
        }
        # 获取任务结果，处理超时逻辑
        for i, future in enumerate(futures):
            try:
                # 设置结果返回的超时时间。与 ProcessPoolExecutor 不同，Ray 在这里通过 ray.get 的 timeout 参数控制
                task_result = ray.get(future, timeout=self.timeout_seconds)

                # 标准化 task_result 的格式
                if isinstance(task_result, dict):
                    assert 'extra_info' in task_result, f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    score_result = task_result
                    # 如果计算结果未过滤，确保正确标记
                    # if "is_filter" not in task_result["extra_info"]:
                    #     score_result["extra_info"].update({"is_filter": 0})
                # elif isinstance(task_result, (int, float)):  # 处理标量返回结果
                #     score_result = copy.deepcopy(default_fail_score) 
                #     # score_result = {"score": float(task_result), "extra_info": {"is_filter": 0}}
                #     score_result['score'] = float(task_result)
                else:
                    print(f"Unexpected task_result type for item {i}: {type(task_result)}. Using default score. Result: {task_result}")
                    ray.cancel(future, force=True)
                    score_result = default_fail_score
            except GetTimeoutError:
                print(f"Timeout processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'). Using default score.")
                score_result = default_fail_score
            except Exception as e:
                print(f"Error processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'): {e}")
                import traceback
                traceback.print_exc()
                ray.cancel(future, force=True)
                score_result = default_fail_score

            # 存储最终得分
            scores[i] = float(score_result.get('score', self.default_negative_score))  # 确保 score 是 float 类型

            if isinstance(score_result, dict):
                # Store the information including original reward
                for key, value in score_result.items():
                    if key == 'extra_info' and isinstance(score_result['extra_info'], dict):
                        for sub_key, sub_value in score_result['extra_info'].items():
                            if sub_key not in extra_info_dict:
                                # 初始化列表（例如默认值 0.0）以匹配所有项
                                extra_info_dict[sub_key] = [0.0] * len(solution_strs)
                            extra_info_dict[sub_key][i] = sub_value
                    else:
                        if key not in extra_info_dict:
                            # 初始化列表（例如默认值 0.0）以匹配所有项
                            extra_info_dict[key] = [0.0] * len(solution_strs)
                        extra_info_dict[key][i] = value
                    
        return scores, extra_info_dict

    def call_with_ray(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # reward_extra_info = defaultdict(list)
        
        sequences_strs = self.tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_info = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]

        already_print_data_sources = {}

        loop_start_time = time.time()
        scores, reward_extra_info = self.math_compute_score_parallel_with_ray(data_sources, sequences_strs, ground_truths, extra_info)
        
        # align with naive calling
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            reward = scores[i]
            reward_tensor[i, valid_response_length - 1] = reward
            
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and random.random()<0.01:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(scores[i], dict):
                    for key, value in scores[i].items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", scores[i])
                            
        loop_end_time = time.time()
        print(f"[Verify Time] {loop_end_time - loop_start_time:.2f} seconds for {len(data)} samples")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
 
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        
        # if we need to run with ray multitask, use the new call function
        if self.multi_proc:
            return self.call_with_ray(data, return_dict)
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
