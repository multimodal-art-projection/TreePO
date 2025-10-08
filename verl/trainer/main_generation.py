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
Generate responses given a dataset of prompts
"""

import os
import time 
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint
from collections import defaultdict

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.device import is_cuda_available


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    # shuffle the merged benchmarks to balance, but keep the df index
    dataset = dataset.sample(frac=1, random_state=42)
    
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    # wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda" if is_cuda_available else "npu")
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda")
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    if 'uuid' in dataset:
        print('using uuid as index')
        uuid_lst = dataset['uuid'].to_numpy()
    else:
        uuid_lst = None
    
    output_str_lst = defaultdict(list)
    output_len_lst = defaultdict(list)
    # non_tensor_batch["dp_batch_total_time"] = np.array([end_time - start_time for _ in range(len(prompts))], dtype=object)
    # non_tensor_batch["dp_batch_n_trajectory"] = np.array([len(response) for _ in range(len(prompts))], dtype=object) # 保留这个数据，消除 dp 分配的影响
    # non_tensor_batch["dp_batch_response_len"] 
    output_batch_idx_lst = defaultdict(list)
    output_batch_total_time_lst = defaultdict(list)
    output_batch_trajectory_lst = defaultdict(list)
    output_response_len_lst = defaultdict(list)


    total_time_cost = 0.0
    
    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        
        # add index for tracking multiple generations
        if uuid_lst is None:
            data.non_tensor_batch['index'] = np.arange(batch_idx * config_batch_size, batch_idx * config_batch_size + data.batch.batch_size[0])
        else:
            # 使用 UUID 列作为索引，后续可以统计每个 UUID 的生成结果
            data.non_tensor_batch['index'] = uuid_lst[batch_idx * config_batch_size : batch_idx * config_batch_size + data.batch.batch_size[0]]
        
        if hasattr(config.rollout, "infer_mode") and config.rollout.infer_mode == "tree":
            print('skip repeating for tree generation, use gen-func internal sampling')
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)   
            # the last is pad
            num_padded = data_padded.batch.batch_size[0] - data.batch.batch_size[0]
            if num_padded > 0:
                data_padded.non_tensor_batch['index'][-num_padded:] = -1
            # the last is pad
            num_padded = data_padded.batch.batch_size[0] - data.batch.batch_size[0]
            if num_padded > 0:
                data_padded.non_tensor_batch['index'][-num_padded:] = -1

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}/{num_batch}] Start to generate. There are {data_padded.batch.batch_size[0]} samples ({num_padded} samples are padded).")

            ts = time.time()
            output_padded = wg.generate_sequences(data_padded)
            te = time.time()
            print(f"[{batch_idx + 1}/{num_batch}] Finish generation. Time cost: {(te - ts) / 60:.1f} m.")

            total_time_cost += (te - ts)
            
            for i in range(len(output_padded)):
                data_item = output_padded[i]
                index = data_item.non_tensor_batch['index']
                if index == -1:
                    continue

                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                output_str_lst[index].append(response_str)
                output_len_lst[index].append(valid_response_length.item())

                # 保存额外信息，用于统计：
                output_batch_idx_lst[index].append(batch_idx)
                output_batch_total_time_lst[index].append(data_item.non_tensor_batch['dp_batch_total_time'])
                output_batch_trajectory_lst[index].append(data_item.non_tensor_batch['dp_batch_n_trajectory'])
                output_response_len_lst[index].append(data_item.non_tensor_batch['dp_batch_response_len'])

        else:
            data = data.repeat(config.data.n_samples, interleave=True)  # [1 2] -> [1 1 1 1 2 2 2 2]
            # data = data.repeat(config.data.n_samples, interleave=False)   # [1 2] -> [1 2 1 2 1 2 1 2]  -> more balanced across GPUs， # 这个 trick 用上了对我们不公平

            # pad data to fullfill the divisible requirement
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)   

            # the last is pad
            num_padded = data_padded.batch.batch_size[0] - data.batch.batch_size[0]
            if num_padded > 0:
                data_padded.non_tensor_batch['index'][-num_padded:] = -1

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}/{num_batch}] Start to generate. There are {data_padded.batch.batch_size[0]} samples ({num_padded} samples are padded).")

            ts = time.time()
            output_padded = wg.generate_sequences(data_padded)
            te = time.time()
            print(f"[{batch_idx + 1}/{num_batch}] Finish generation. Time cost: {(te - ts) / 60:.1f} m.")

            total_time_cost += (te - ts)

            output = unpad_dataproto(output_padded, pad_size=pad_size)

            for i in range(len(data)):
                data_item = output[i]
                index = data_item.non_tensor_batch['index']
                if index == -1:
                    continue

                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                output_str_lst[index].append(response_str)
                output_len_lst[index].append(valid_response_length.item())

                # 保存额外信息，用于统计：
                output_batch_idx_lst[index].append(batch_idx)
                output_batch_total_time_lst[index].append(data_item.non_tensor_batch['dp_batch_total_time'])
                output_batch_trajectory_lst[index].append(data_item.non_tensor_batch['dp_batch_n_trajectory'])
                output_response_len_lst[index].append(data_item.non_tensor_batch['dp_batch_response_len'])

    output_str_lst = list(output_str_lst.values())   # (n_data, n_samples)
    output_len_lst = list(output_len_lst.values())   # (n_data, n_samples)

    # 转换
    output_batch_idx_lst = list(output_batch_idx_lst.values())
    output_batch_total_time_lst = list(output_batch_total_time_lst.values())
    output_batch_trajectory_lst = list(output_batch_trajectory_lst.values())
    output_response_len_lst = list(output_response_len_lst.values())

    # add to the data frame
    dataset["responses"] = output_str_lst
    dataset["response_length"] = output_len_lst
    
    dataset["total_generation_time"] = [total_time_cost for _ in range(len(dataset))]
    
    dataset["dp_batch_idx"] = output_batch_idx_lst
    dataset["dp_batch_total_time"] = output_batch_total_time_lst
    dataset["dp_batch_trajectory"] = output_batch_trajectory_lst
    dataset["dp_batch_response_len"] = output_response_len_lst


    # sort the dataset back by index
    dataset = dataset.sort_index()
        
    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
