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
import time
import ray

from typing import List, Dict
from ray.exceptions import GetTimeoutError  # 用于处理超时
from verl.workers.reward_manager.utils import reward_func_timeout_ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score

import numpy as np
import copy

def compute_one_item(args):
    (
        idx,
        prompt_str,
        response_str,
        valid_response_length,
        ground_truth,
        data_source,
        extra_info,
        max_resp_len,
        overlong_buffer_cfg,
        compute_score,
        eos_token,
    ) = args

    # Remove EOS if needed (should already be done in main process, but safe to check)
    if response_str.endswith(eos_token):
        response_str = response_str[: -len(eos_token)]

    result = compute_score(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

    reward_extra_info = defaultdict(list)
    if isinstance(result, dict):
        score = result["score"]
        for key, value in result.items():
            reward_extra_info[key].append(value)
    else:
        score = result

    reward = score

    if overlong_buffer_cfg and overlong_buffer_cfg.enable:
        overlong_buffer_len = overlong_buffer_cfg.len
        expected_len = max_resp_len - overlong_buffer_len
        exceed_len = valid_response_length - expected_len
        overlong_penalty_factor = overlong_buffer_cfg.penalty_factor
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        reward += overlong_reward
        if overlong_buffer_cfg.log:
            reward_extra_info["overlong_reward"].append(overlong_reward)
            reward_extra_info["overlong"].append(overlong_reward < 0)

    return (
        idx,
        valid_response_length,
        reward,
        reward_extra_info,
        prompt_str,
        response_str,
        ground_truth,
        data_source,
        result,
    )
    
@ray.remote
def compute_one_item_ray(args):
    return compute_one_item(args)
    
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        multi_proc=True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.multi_proc = multi_proc
        
        self.timeout_seconds = 5

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    # 修改原始方法，使用 Ray
    def math_compute_score_parallel_with_ray(
        self, 
        data_sources, solution_strs, ground_truths, extra_infos,
        # first_rollout_rewards, second_reward_rates
    ):
        scores: List[float] = [-1.0] * len(solution_strs)
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
                self.compute_score, self.timeout_seconds, data_source, solution_str, ground_truth, extra_info
            )
            futures.append(future)

        # default_fail_score = {"score": 0.0, "extra_info": {"is_filter": 1}}  # Default on error which should be filtered
        default_fail_score = {
            "score": -1.0, "acc": 0.0, 
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
            scores[i] = float(score_result.get('score', -1.0))  # 确保 score 是 float 类型

            # # 如果存在 extra_info，收集它
            # if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
            #     for key, value in score_result['extra_info'].items():
            #         if key not in extra_info_dict:
            #             # 初始化列表（例如默认值 0.0）以匹配所有项
            #             extra_info_dict[key] = [0.0] * len(solution_strs)
            #         extra_info_dict[key][i] = value

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
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
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

        # # Use Ray for multi-processing if enabled
        # ray_futures = [compute_one_item_ray.remote(args) for args in decoded_data]
        # results = ray.get(ray_futures)
        # for (
        #     idx,
        #     valid_response_length,
        #     reward,
        #     reward_extra_info_item,
        #     prompt_str,
        #     response_str,
        #     ground_truth,
        #     data_source,
        #     result,
        # ) in results:
        #     reward_tensor[idx, valid_response_length - 1] = reward
        #     for key, vals in reward_extra_info_item.items():
        #         reward_extra_info[key].extend(vals)

        #     if data_source not in already_print_data_sources:
        #         already_print_data_sources[data_source] = 0

        #     if already_print_data_sources[data_source] < self.num_examine:
        #         already_print_data_sources[data_source] += 1
        #         print("[prompt]", prompt_str)
        #         print("[response]", response_str)
        #         print("[ground_truth]", ground_truth)
        #         if isinstance(result, dict):
        #             for key, value in result.items():
        #                 print(f"[{key}]", value)
        #         else:
        #             print("[score]", reward)
        # # Pre-decode all prompt/response strings and collect necessary info
        # decoded_data = []
        # for idx in range(len(data)):
        #     data_item = data[idx]  # DataProtoItem

        #     prompt_ids = data_item.batch["prompts"]
        #     prompt_length = prompt_ids.shape[-1]
        #     valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch["responses"]
        #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]

        #     prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        #     response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        #     eos_token = self.tokenizer.eos_token
        #     if response_str.endswith(eos_token):
        #         response_str = response_str[: -len(eos_token)]

        #     ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        #     data_source = data_item.non_tensor_batch[self.reward_fn_key]
        #     extra_info = data_item.non_tensor_batch.get("extra_info", None)

        #     decoded_data.append(
        #         (
        #             idx,
        #             prompt_str,
        #             response_str,
        #             valid_response_length,
        #             ground_truth,
        #             data_source,
        #             extra_info,
        #             self.max_resp_len,
        #             self.overlong_buffer_cfg,
        #             self.compute_score,
        #             eos_token,
        #         )
        #     )
        # if self.multi_proc:   
        #     ray_futures = []
        #     for args in decoded_data:
        #         (
        #             idx,prompt_str,response_str,valid_response_length,
        #             ground_truth,data_source,extra_info, max_resp_len,
        #             overlong_buffer_cfg, compute_score, eos_token,
        #         ) = args
        #         ray_futures.append(
        #             reward_func_timeout_ray.remote(
        #                 self.compute_score,
        #                 self.timeout_seconds,
        #                 data_source=data_source,
        #                 solution_str=response_str,
        #                 ground_truth=ground_truth,
        #                 extra_info=extra_info,
        #             )
        #         )
        #     # default_fail_score = {"score": 0.0, "extra_info": {"is_filter": 1}}
        #     default_fail_score = {"score": -1.0, "acc": 0.0, "pred": 'Timeout Placeholder',}
        #     for idx, future in enumerate(ray_futures):
        #         try:
        #             # 设置结果返回的超时时间。与 ProcessPoolExecutor 不同，Ray 在这里通过 ray.get 的 timeout 参数控制
        #             task_result = ray.get(future, timeout=self.timeout_seconds)
        #             # 标准化 task_result 的格式
        #             if isinstance(task_result, dict):
        #                 # assert 'extra_info' in task_result, f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
        #                 score_result = task_result
        #             elif isinstance(task_result, (int, float)):  # 处理标量返回结果
        #                 score_result = {"score": float(task_result)}
        #             else:
        #                 print(f"Unexpected task_result type for item {idx}: {type(task_result)}. Using default score. Result: {task_result}")
        #                 ray.cancel(future, force=True)
        #                 score_result = default_fail_score
        #         except GetTimeoutError:
        #             print(f"Timeout processing item {idx} (gold='{str(decoded_data[idx][4])[:50]}...', pred='{str(decoded_data[idx][2])[:50]}...'). Using default score.")
        #             score_result = default_fail_score
        #         except Exception as e:
        #             print(f"Error processing item {idx} (gold='{str(decoded_data[idx][4])[:50]}...', pred='{str(decoded_data[idx][2])[:50]}...'): {e}")
        #             import traceback
        #             traceback.print_exc()
        #             ray.cancel(future, force=True)
        #             score_result = default_fail_score
            
        #         (
        #             _,prompt_str,response_str,valid_response_length,
        #             ground_truth,data_source,extra_info, max_resp_len,
        #             overlong_buffer_cfg, compute_score, eos_token,
        #         ) = decoded_data[idx]
                
        #         reward = score_result["score"]
        #         for key, value in score_result.items():
        #             reward_extra_info[key].append(value)
        #         reward_tensor[idx, valid_response_length - 1] = reward
        #         if overlong_buffer_cfg and overlong_buffer_cfg.enable:
        #             overlong_buffer_len = overlong_buffer_cfg.len
        #             expected_len = max_resp_len - overlong_buffer_len
        #             exceed_len = valid_response_length - expected_len
        #             overlong_penalty_factor = overlong_buffer_cfg.penalty_factor
        #             overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        #             reward += overlong_reward
        #             if overlong_buffer_cfg.log:
        #                 reward_extra_info["overlong_reward"].append(overlong_reward)
        #                 reward_extra_info["overlong"].append(overlong_reward < 0)
        # else:
        #     for args in decoded_data:
        #         (
        #             idx,
        #             prompt_str,
        #             response_str,
        #             valid_response_length,
        #             ground_truth,
        #             data_source,
        #             extra_info,
        #             max_resp_len,
        #             overlong_buffer_cfg,
        #             compute_score,
        #             eos_token,
        #         ) = args

        #         # Use the same compute_one_item logic
        #         (
        #             idx,
        #             valid_response_length,
        #             reward,
        #             reward_extra_info_item,
        #             prompt_str,
        #             response_str,
        #             ground_truth,
        #             data_source,
        #             result,
        #         ) = compute_one_item(args)
                
        #         reward_tensor[idx, valid_response_length - 1] = reward
        #         for key, vals in reward_extra_info_item.items():
        #             reward_extra_info[key].extend(vals)

        #         if data_source not in already_print_data_sources:
        #             already_print_data_sources[data_source] = 0

        #         if already_print_data_sources[data_source] < self.num_examine:
        #             already_print_data_sources[data_source] += 1
        #             print("[prompt]", prompt_str)
        #             print("[response]", response_str)
        #             print("[ground_truth]", ground_truth)
        #             if isinstance(result, dict):
        #                 for key, value in result.items():
        #                     print(f"[{key}]", value)
        #             else:
        #                 print("[score]", reward)
        
        # ray calling v2
        scores, reward_extra_info = self.math_compute_score_parallel_with_ray(data_sources, sequences_strs, ground_truths, extra_info)
        # align with DAPO
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
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

        loop_end_time = time.time()
        print(f"[Verify Time] {loop_end_time - loop_start_time:.2f} seconds for {len(data)} samples")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
