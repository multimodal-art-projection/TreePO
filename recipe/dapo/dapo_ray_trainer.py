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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask, pad_dataproto_to_divisor, unpad_dataproto

from recipe.treepo.vllm_rollout_tree import (
    _repeat_by_indices,
)

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                
                gen_batch.non_tensor_batch["uid"] = deepcopy(new_batch.non_tensor_batch["uid"])
                
                def sinusoid_ascending_scheduler(step: int, min_temp: float, max_temp: float, total_steps: int) -> float:
                    """
                    Sinusoid ascending temperature scheduler.
                    """
                    return (min_temp+max_temp)/2 + (max_temp - min_temp) * np.sin(step / total_steps * np.pi - np.pi/2) / 2
                def sinusoid_descending_scheduler(step: int, min_temp: float, max_temp: float, total_steps: int) -> float:
                    """
                    Sinusoid ascending temperature scheduler.
                    """
                    return (min_temp+max_temp)/2 + (max_temp - min_temp) * np.sin(step / total_steps * np.pi + np.pi/2) / 2
                # use meta info to control the divergenc temperature with scheduler
                if self.config.actor_rollout_ref.rollout.get("infer_mode", None) == "tree" and \
                    self.config.actor_rollout_ref.rollout.get("div_temp_scheduler", None) is not None:
                    scheduler_type = self.config.actor_rollout_ref.rollout.div_temp_scheduler.scheduler_type
                    if scheduler_type == "sinusoid_ascending":
                        gen_batch.meta_info['logprob_div_temperature'] = sinusoid_ascending_scheduler(
                            step=self.global_steps,
                            min_temp=self.config.actor_rollout_ref.rollout.div_temp_scheduler.get("min_temp", 0.5),
                            max_temp=self.config.actor_rollout_ref.rollout.div_temp_scheduler.get("max_temp", 10.0),
                            total_steps=self.total_training_steps
                        )
                    elif scheduler_type == "sinusoid_descending":
                        gen_batch.meta_info['logprob_div_temperature'] = sinusoid_descending_scheduler(
                            step=self.global_steps,
                            min_temp=self.config.actor_rollout_ref.rollout.div_temp_scheduler.get("min_temp", 0.5),
                            max_temp=self.config.actor_rollout_ref.rollout.div_temp_scheduler.get("max_temp", 10.0),
                            total_steps=self.total_training_steps
                        )
                    elif scheduler_type == "constant":
                        gen_batch.meta_info['logprob_div_temperature'] = self.config.actor_rollout_ref.rollout.get("tree_logprob_div_temperature", 1.0)
                    else:
                        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Please check your config.")
                
                    metrics["actor/div_temperature"] = gen_batch.meta_info['logprob_div_temperature']
                    
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    if self.config.actor_rollout_ref.rollout.get("infer_mode", None) == "tree":
                        uid2original_idx = {}
                        for original_idx, uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            uid2original_idx[uid]=original_idx
                        global_duplicated_idxs = []
                        for uid in gen_batch_output.non_tensor_batch["uid"]:
                            global_duplicated_idxs.append(uid2original_idx[uid])
                        new_batch = new_batch[global_duplicated_idxs] # uuid 和 root node 一一对应
                    else:
                        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)
                        
                        prompt_bsz = self.config.data.train_batch_size
                        # 在这里就直接 truncate 多出来的 prompts
                        if num_prompt_in_batch > prompt_bsz:
                            n_additional = num_prompt_in_batch - prompt_bsz
                            print(f"{num_prompt_in_batch=} > {prompt_bsz=}, remove the {n_additional=} prompts...")
                            kept_prompt_uids = kept_prompt_uids[:-n_additional]
                            num_prompt_in_batch = prompt_bsz
                        
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch （即去掉比 bsz 多余的 trajectory）
                            if self.config.actor_rollout_ref.rollout.get("infer_mode", None) == "tree":
                                # 由于 tree infer 可能存在 trajectory 数量 不一致的情况，不能直接截断
                                pass
                            else:
                                traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                batch = batch[:traj_bsz]

                    # === Updating ===
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.actor_rollout_ref.rollout.get("infer_mode", None) == "tree" \
                        and self.config.actor_rollout_ref.rollout.tree_total_budget_policy != "by_response_traj":
                        batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                        print(f"Pad size: {pad_size}, new batch size: {len(batch)}, this may cause different behavior")
                        
                    # leveraging "uid" to do subgroup re-matching
                    if self.config.actor_rollout_ref.actor.get("tree_subgroup_opt", -1) > 0:
                        subgroup_depth = self.config.actor_rollout_ref.actor.tree_subgroup_opt
                        new_uuids = []
                        for traj_idx, prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
                            # 按深度取出树的 path prefix
                            subgroup_id = "#".join(batch.non_tensor_batch["tree_idx"][traj_idx].split('/')[1:subgroup_depth+1])
                            new_uuids.append(prompt_uid + "#" + subgroup_id)
                        batch.non_tensor_batch["uid"] = np.array(new_uuids, dtype=object)
                            
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        if self.config.algorithm.adv_estimator in [AdvantageEstimator.TREE_SEG_BASELINE, AdvantageEstimator.SEGPO]:
                            batch.meta_info["step_max_token"] = self.config.actor_rollout_ref.rollout.tree_max_token_per_step
                            if hasattr(self.config.algorithm, "adv_estimator_param"):
                                batch.meta_info["segtree_adv_eta"] = self.config.algorithm.adv_estimator_param.get("segtree_adv_eta", -1.0)
                        if self.config.algorithm.adv_estimator in [AdvantageEstimator.TREE_REINFORCE_PLUS_PLUS_BASELINE]:
                            if hasattr(self.config.algorithm, "adv_estimator_param"):
                                batch.meta_info["subtree_std_ds"] = self.config.algorithm.adv_estimator_param.get("subtree_std_ds", False) # dynamic sampling for the no-comparison subtrees
                                batch.meta_info["adv_weight_strategy"] = self.config.algorithm.adv_estimator_param.get("adv_weight_strategy", "group_size")
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                
                if self.config.actor_rollout_ref.rollout.get("infer_mode", None) == "tree" \
                    and self.config.actor_rollout_ref.rollout.tree_total_budget_policy != "by_response_traj" \
                    and pad_size > 0:
                    # 这里要 pad 回来，否则会导致后面的 n-traj_per_prompt 等 metric 计算错误
                    batch = unpad_dataproto(batch, pad_size=pad_size)

                # 还原 UUID （取 36），去掉后缀
                if self.config.actor_rollout_ref.actor.get("tree_subgroup_opt", -1) > 0:
                    original_uuids = [prompt_uid[:36] for prompt_uid in batch.non_tensor_batch["uid"]]
                    batch.non_tensor_batch["uid"] = np.array(original_uuids, dtype=object)
                    
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing
                
                # 记录 trajectory 数量
                prompt_uid2count = defaultdict(float)
                for prompt_uid in batch.non_tensor_batch["uid"]:
                    prompt_uid2count[prompt_uid] += 1.0
                metrics["train/n_unique_prompt"] = len(prompt_uid2count.keys())
                metrics["train/n-traj_per_prompt/avg"] = float(np.mean(list(prompt_uid2count.values())))
                metrics["train/n-traj_per_prompt/min"] = float(np.min(list(prompt_uid2count.values())))
                metrics["train/n-traj_per_prompt/max"] = float(np.max(list(prompt_uid2count.values())))
                metrics["train/n-traj_per_prompt/std"] = float(np.std(list(prompt_uid2count.values())))
                
                metrics["timing_s/gen_avg_traj"] = metrics["timing_s/gen"]/sum(list(prompt_uid2count.values()))
        
                if self.config.actor_rollout_ref.rollout.get("infer_mode", None) == "tree":
                    prompt_uid2fallback_count = defaultdict(float)
                    prompt_uid2nonfallback_count = defaultdict(float)
                    prompt_uid2depth = defaultdict(list)
                    for traj_idx, prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
                        if prompt_uid not in prompt_uid2fallback_count:
                            prompt_uid2fallback_count[prompt_uid] = batch.non_tensor_batch["fallback_counter"][traj_idx]
                        if prompt_uid not in prompt_uid2nonfallback_count:
                            prompt_uid2nonfallback_count[prompt_uid] = batch.non_tensor_batch["n_finished_traj_at_first_fallback"][traj_idx]

                        # 取出树的深度
                        traj_tree_depth = len(batch.non_tensor_batch["tree_idx"][traj_idx].split('/')) - 1
                        prompt_uid2depth[prompt_uid].append(traj_tree_depth)
                    
                    # 统计平均完成多少条 traj 以后，开始产生 fallback（排除没有做 fallback 的 prompt
                    metrics["train/n-finished-traj_first_fallback/avg"] = float(np.mean([v for v in prompt_uid2nonfallback_count.values() if v > 0]))
                    # 统计每个 prompt 平均有多少条 non-fallback 产生的 traj
                    metrics["train/n-traj_non-fallback/avg"] = float(np.mean([prompt_uid2nonfallback_count[prompt_uid] if prompt_uid2nonfallback_count[prompt_uid] > 0 else prompt_uid2count[prompt_uid] for prompt_uid in prompt_uid2nonfallback_count ]))
                    # 统计有 fallback 的 uid 比例
                    metrics["train/prompt_fallback_ratio"] = len([v for v in prompt_uid2nonfallback_count.values() if v > 0])/len(prompt_uid2fallback_count)
                    # 统计平均每个 prompt uid 有多少个 fallback
                    metrics["train/n-fallback_per_prompt/avg"] = float(np.mean(list(prompt_uid2fallback_count.values())))
                    metrics["train/n-fallback_per_prompt/min"] = float(np.min(list(prompt_uid2fallback_count.values())))
                    metrics["train/n-fallback_per_prompt/max"] = float(np.max(list(prompt_uid2fallback_count.values())))
                    metrics["train/n-fallback_per_prompt/std"] = float(np.std(list(prompt_uid2fallback_count.values())))    
                    
                    metrics['train/traj_finish_step/avg'] = float(np.mean(batch.non_tensor_batch['traj_finish_step']))
                    metrics['train/traj_finish_step/min'] = float(np.min(batch.non_tensor_batch['traj_finish_step']))
                    metrics['train/traj_finish_step/max'] = float(np.max(batch.non_tensor_batch['traj_finish_step']))
                    metrics['train/traj_finish_step/std'] = float(np.std(batch.non_tensor_batch['traj_finish_step']))  
                    
                    # 统计每个 prompt 的深度
                    metrics["tree/prompt_tree_mean_depth/avg"] = float(np.mean([np.mean(depths) for _, depths in prompt_uid2depth.items()]))
                    metrics["tree/prompt_tree_mean_depth/min"] = float(np.min([np.mean(depths) for _, depths in prompt_uid2depth.items()]))
                    metrics["tree/prompt_tree_mean_depth/max"] = float(np.max([np.mean(depths) for _, depths in prompt_uid2depth.items()]))
                    metrics["tree/prompt_tree_mean_depth/std"] = float(np.std([np.mean(depths) for _, depths in prompt_uid2depth.items()]))
                    
                    # 记录各个深度的比例
                    all_depths = [d for _, depths in prompt_uid2depth.items() for d in depths]
                    for depth in set(all_depths):
                        metrics[f'tree/depth-{depth}_ratio'] = all_depths.count(depth)/len(all_depths)
                    
                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
