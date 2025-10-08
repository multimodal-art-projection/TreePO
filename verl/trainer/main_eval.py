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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from collections import defaultdict, Counter

from verl.utils.fs import copy_to_local


def get_custom_reward_fn(config):
    import importlib.util
    import os
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}'") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data, compute_type="mean"):
    ground_truth = reward_data["ground_truth"]
    
    # YL: MINERVA 的  GT 似乎有问题
    if isinstance(ground_truth, list) or isinstance(ground_truth, np.ndarray):
        if len(ground_truth)!=1:
            print(f"[WARNING] multiple ground truths:\n{ground_truth}\n using the last one {ground_truth[-1]}")
        ground_truth = ground_truth[-1]
    if isinstance(ground_truth, int) or isinstance(ground_truth, float): 
        ground_truth = str(ground_truth)
        ground_truth = ground_truth.strip().rstrip()
    
    verify_list = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    score_lst = verify_list
    if compute_type=="mean":
        if type(score_lst[0]) == dict: 
            score_lst = [s["score"] for s in score_lst]
        # print("raw score:", score_lst)
        score_lst = [1 if s>0 else 0 for s in score_lst]
        # print(score_lst)
        if np.mean(score_lst)==0:
            print("groundtruth:", ground_truth, "preds:",  [x["extra_info"]["pred"] for x in verify_list])
        # raise 
        return data_source, np.mean(score_lst)
    elif compute_type=="major":
        # 算major必须返回的是dict
        assert type(score_lst[0]) == dict
        answer_list = [s['extra_info']['pred'] for s in score_lst]
        score_lst = [s["score"] for s in score_lst]
        score_lst = [1 if s==1 else 0 for s in score_lst]
        cnt = Counter(answer_list)
        majority = cnt.most_common(1)[0][0]
        groups = defaultdict(list)
        for idx, pred in enumerate(answer_list):
            groups[pred].append(idx)
        majority_idx = groups[majority][0]
        # print(f"majority:{majority}: {cnt.most_common(1)[0][1]} : {score_lst}")
        return data_source, score_lst[majority_idx]
    elif compute_type=="major_v2":
        # 使用 bootstrap_metric 和 calc_maj_val 计算 major accuracy
        assert type(score_lst[0]) == dict
        answer_list = [s['extra_info']['pred'] for s in score_lst]
        score_lst = [s["score"] for s in score_lst]
        score_lst = [1 if s==1 else 0 for s in score_lst]
        
        vote_data = [{"pred": pred, "val": score} for pred, score in zip(answer_list, score_lst)]
        
        # 使用 bootstrap_metric 计算 major accuracy
        from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val
        from functools import partial
        
        # rollout n数量
        n_resps = len(vote_data)
        # 不同的major数量
        if n_resps > 1:
            # ns = []
            # n = 2
            # while n < n_resps:
            #     ns.append(n)
            #     n *= 2
            # ns.append(n_resps)
            
            # 使用所有样本计算 major accuracy（与训练时保持一致）
            [(maj_mean, maj_std)] = bootstrap_metric(
                data=vote_data,
                subset_size=n_resps,
                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                n_bootstrap=1000,
                seed=42,
            )
            return data_source, maj_mean
    else:
        raise NameError("compute_type only mean or major")


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get('use_shm', False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_reward_major = defaultdict(list)
    compute_score = get_custom_reward_fn(config)
    if compute_score is None:
        from verl.utils.reward_score import default_compute_score
        compute_score = default_compute_score

    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]
    # remote_tasks_major = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i], "major") for i in range(total)]
    remote_tasks_major = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i], "major_v2") for i in range(total)]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)
    
    # cal major score
    with tqdm(total=total) as pbar:
        while len(remote_tasks_major) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks_major = ray.wait(remote_tasks_major)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward_major[data_source].append(score)
                pbar.update(1)

    import os
    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f'{os.environ["model_path"]}/test_score/{data_source}'] = float(np.mean(rewards))
    
    for data_source, rewards in data_source_reward_major.items():
        metric_dict[f'{os.environ["model_path"]}/test_score_major/{data_source}'] = float(np.mean(rewards))

    if 'total_generation_time' in dataset.columns:
        total_generation_time = dataset['total_generation_time'].mean()
        metric_dict['total_generation_time'] = float(total_generation_time)
        # print(f"Total generation time: {total_generation_time:.2f} seconds")
    
    # 单独每个 dp 统计时间以后再平均，更精准
    if "dp_batch_idx" in dataset.columns:
        # 用 time 做 dp 的索引
        dp_time_lst = set([t for time_lst in dataset['dp_batch_total_time'] for t in time_lst])
        assert len(dp_time_lst) % 8 == 0, "肯定是 8*Node 个数字"
        
        dp_time2tokens = defaultdict(list)
        dp_time2traj = defaultdict(int)
            
        for row_index, row in dataset.iterrows():
            # dp_time, dp_n_traj 在同一个 dp 内是一样的，只有 response_len 有差别
            for dp_time, dp_n_traj, response_len in zip(row['dp_batch_total_time'], row['dp_batch_trajectory'], row["dp_batch_response_len"]):
                dp_time2tokens[dp_time].append(int(response_len))
                if dp_time not in dp_time2traj:
                    dp_time2traj[dp_time] = int(dp_n_traj)
                    
        assert len(list(dp_time2traj.keys()))  % 8 == 0, "肯定是 8*Node 个数字"
        
        # 近似单卡的速度，所以每个 dp 内先做平均
        dp_wise_token_time = []
        dp_wise_traj_time = []
        for dp_time, resp_len_lst in dp_time2tokens.items():
            dp_wise_token_time.append(np.sum(resp_len_lst)/dp_time) # 计算在一个 dp 内，单位时间生产了多少 token
        for dp_time, n_traj in dp_time2traj.items():
            dp_wise_traj_time.append(n_traj/dp_time)
        
        n_token_per_second = float(np.mean(dp_wise_token_time))
        n_traj_per_second = float(np.mean(dp_wise_traj_time))
        
        metric_dict['n_token_per_second'] = n_token_per_second
        metric_dict['n_traj_per_second'] = n_traj_per_second

        # 计算一共用了多少 gpu seconds
        total_gpu_seconds = sum(dp_time2tokens.keys())
        metric_dict['total_GPU_seconds'] = float(total_gpu_seconds)

    print(metric_dict)


if __name__ == "__main__":
    main()
