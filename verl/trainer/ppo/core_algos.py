# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

import copy

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_tree_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6, 
    remove_no_std_subgroup=False, adv_weight_strategy='group_size',
    ):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    traj_idx2adv_list = {traj_idx : [] for traj_idx, _ in enumerate(index)} # {traj_idx: [{"score": tensor, "group_size": int}, ...] } 记录每个叶子节点在不同 tree (group) 算出来的分数和 group size

    with torch.no_grad():
        # 初始化
        bsz = scores.shape[0]
        max_tree_depth = max([len(uid.split("#"))-1 for uid in index])
        
        # calculate the group-based scores by depth
        # 不需要处理 max_tree_depth 本身，因为是叶子节点
        for depth in range(max_tree_depth): 
            id2score = defaultdict(list)
            id2mean = {} 
            id2std = {}
            
            # 根据深度重新处理，对 id 分类重新分类
            scores_by_depth = scores.clone()
            index_by_depth = ["#".join(uid.split("#")[:depth+1]) for uid in index] # depth==0 时，就是 root node 分组
            
            # 按组收集分数
            for i in range(bsz):
                id2score[index_by_depth[i]].append(scores_by_depth[i])
            # 对每个组里的分数求平均
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(0.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            # 减去平均值
            for i in range(bsz):
                scores_by_depth[i] = scores_by_depth[i] - id2mean[index_by_depth[i]]
            # 加入到叶子节点的组里面
            for i in range(bsz):
                # 只统计相对优势，如果 subgroup 只有自己，则不统计
                if len(id2score[index_by_depth[i]]) > 1 and depth > 0:
                    # skip the sample if no relevant adv
                    if remove_no_std_subgroup and (0.0 == id2std[index_by_depth[i]]):
                        pass
                    else:
                        traj_idx2adv_list[i].append({
                            "score": scores_by_depth[i],
                            "group_size": len(id2score[index_by_depth[i]]),
                        })
                # depth==0 的 reward 必须加！
                elif depth == 0:
                    traj_idx2adv_list[i].append({
                        "score": scores_by_depth[i],
                        "group_size": len(id2score[index_by_depth[i]]),
                    })
            # print(f"{depth=}")
            # print(scores_by_depth)
        # 统计完所有深度的分数，给每个 trajectory 计算加权相对 advantage
        if adv_weight_strategy=='group_size':
            for i in range(bsz):
                total_group_size = sum([item["group_size"] for item in traj_idx2adv_list[i]])
                scores[i] = torch.tensor([item["score"]*item["group_size"]/total_group_size for item in traj_idx2adv_list[i]]).sum()
        elif adv_weight_strategy=='average':
            for i in range(bsz):
                scores[i] = torch.tensor([item["score"] for item in traj_idx2adv_list[i]]).mean()
        else:
            raise NotImplementedError
        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask
        
    return scores, scores

def tree_group_wise_reward(data: Dict[str, float]) -> Dict[str, float]:
    """
    对原始的 leaf reward 进行修正。

    修正逻辑如下：
    1. 对每个中间节点，计算其下所有 leaf node reward 的均值（predecessor reward）和数量（group size）。
    2. 对每个 leaf node，计算其 reward 与其所有上级节点（predecessor）的 predecessor reward 的差值。
    3. 将这些差值根据 group size 进行加权平均，作为此 leaf node 的新 reward。

    Args:
        data: 原始的 trajectory 数据。

    Returns:
        一个新字典，key 是 trajectory 路径，value 是修正后的 reward。
    """
    if not data:
        return {}

    # --- Step 1: 计算所有中间节点的 predecessor reward 和 group size ---
    predecessor_to_leaves = defaultdict(list)
    # 遍历所有 leaf node，将其 reward 添加到其所有上级节点的列表中
    for leaf_path, leaf_reward in data.items():
        parts = leaf_path.split('#')
        # 从 depth=1 的节点开始，收集所有前缀
        for i in range(1, len(parts) + 1):
            predecessor_path = '#'.join(parts[:i])
            predecessor_to_leaves[predecessor_path].append(leaf_reward)

    # 计算每个 predecessor 的 reward (mean) 和 size (len)
    predecessor_info = {}
    for path, rewards in predecessor_to_leaves.items():
        predecessor_info[path] = {
            "reward": sum(rewards) / len(rewards),
            "size": len(rewards)
        }

    # --- Step 2 & 3: 计算每个 leaf node 修正后的 reward ---
    modified_rewards = {}
    for leaf_path, leaf_reward in data.items():
        total_weighted_advantage = 0
        total_weight = 0
        
        parts = leaf_path.split('#')
        # 遍历该 leaf 的所有上级节点（包括它自己）
        for i in range(1, len(parts) + 1):
            predecessor_path = '#'.join(parts[:i])
            
            info = predecessor_info[predecessor_path]
            pred_reward = info["reward"]
            pred_size = info["size"]
            
            # 计算 leaf reward 相对于这个 subgroup 的优势
            advantage = leaf_reward - pred_reward
            
            total_weighted_advantage += advantage * pred_size
            total_weight += pred_size
            
        # 计算加权平均的优势作为新的 reward
        if total_weight > 0:
            modified_rewards[leaf_path] = total_weighted_advantage / total_weight
        else:
            modified_rewards[leaf_path] = 0.0 # Should not happen

    return modified_rewards


def calculate_rewards_iterative(data: Dict[str, float]) -> Dict[str, float]:
    """
    使用迭代（for-loop）和自底向上的方法计算每个节点的 progress reward。

    这种方法的核心思想是“自底向上”。我们首先确定最深的层级，
    然后逐层向上计算父节点的奖励，直到到达 depth=1 的节点。

    Args:
        data: 一个字典，key 为 trajectory 字符串，value 为最终分数。

    Returns:
        一个字典，包含了所有中间节点和叶子节点的 reward。
    """
    if not data:
        return {}

    # 1. 初始化 rewards 字典，它将包含所有叶子节点和中间节点的最终奖励。
    #    同时，找到所有路径中的最大深度。
    rewards = data.copy()
    max_depth = 0
    for path in data:
        # 路径的深度等于 '#' 的数量
        depth = path.count('#')
        if depth > max_depth:
            max_depth = depth

    # 2. 从最深层 (max_depth) 向上迭代到第 2 层。
    #    在每一轮循环中，我们计算 depth = d-1 的节点的 reward。
    #    我们只需要循环到 2，因为我们计算的是 depth=1 节点的奖励，其父节点是 depth=0 (root)，而 root 不需要计算。
    for d in range(max_depth, 1, -1):
        # 用于收集当前深度下，每个父节点对应的所有子节点的 reward
        parent_to_children_rewards = defaultdict(list)

        # 遍历所有已经计算出 reward 的节点
        for path, reward in rewards.items():
            # 我们只关心当前深度 d 的节点，因为我们要用它们来计算上一层的奖励
            if path.count('#') == d:
                # 通过截断路径获取其父节点的路径
                parent_path = '#'.join(path.split('#')[:-1])
                parent_to_children_rewards[parent_path].append(reward)

        # 3. 计算父节点的 reward (即其所有子节点 reward 的平均值)
        for parent_path, child_rewards in parent_to_children_rewards.items():
            if child_rewards:
                mean_reward = sum(child_rewards) / len(child_rewards)
                rewards[parent_path] = mean_reward

    return rewards

def normalize_sibling_rewards(all_node_rewards: Dict[str, float]) -> Dict[str, float]:
    """
    对每个节点的 reward，根据其兄弟节点（siblings）的分布进行标准化。
    标准化公式：reward' = (reward - mean(siblings_rewards)) / std(siblings_rewards)
    
    此操作在所有节点的 reward 计算完毕后执行。

    Args:
        all_node_rewards: 一个字典，包含了所有节点（叶子节点和中间节点）的 reward。

    Returns:
        一个新字典，其中所有存在兄弟节点的 node reward 都已被标准化。
    """
    if not all_node_rewards:
        return {}
    
    # 1. 首先，将所有节点按其父节点进行分组
    parent_to_children = defaultdict(list)
    for path in all_node_rewards:
        if '#' in path: # 根节点没有父节点，不处理
            parent_path = '#'.join(path.split('#')[:-1])
            parent_to_children[parent_path].append(path)

    # 创建一个结果字典的副本，我们将在此基础上进行修改
    normalized_rewards = all_node_rewards.copy()

    # 2. 遍历每个分组（即每个 sibling 集合）
    for parent_path, sibling_paths in parent_to_children.items():
        # 如果节点没有兄弟（即只有一个孩子），则跳过
        if len(sibling_paths) < 2:
            continue
            
        # 3. 计算该组兄弟节点 reward 的均值和标准差
        sibling_rewards = [all_node_rewards[p] for p in sibling_paths]
        
        mean_reward = np.mean(sibling_rewards)
        std_reward = np.std(sibling_rewards)
        
        # 4. 对该组的每个节点进行标准化
        for path in sibling_paths:
            original_reward = all_node_rewards[path]
            
            # 如果标准差为0（所有兄弟节点reward都一样），则标准化后的值为0
            if np.isclose(std_reward, 0):
                normalized_rewards[path] = 0.0
            else:
                normalized_rewards[path] = (original_reward - mean_reward) / std_reward
                
    return normalized_rewards

def calculate_segment_advantage(data: Dict[str, float], adv_eta=-1.0, seg_node_subling_norm=False) -> Dict[str, List[float]]:
    """
    计算每个 trajectory 中，每个节点相对于其父节点的 advantage。
    Advantage(node) = Reward(node) - Reward(parent)

    Args:
        data: 原始的 trajectory 数据。

    Returns:
        一个字典，key 是完整的 trajectory 路径，value 是一个列表，
        包含了该路径上从 depth=1 开始每个节点的 advantage 值。
    """
    # 1. 首先，使用现有函数计算出所有节点的 reward
    all_node_rewards = calculate_rewards_iterative(data)
    # 如果是 SPO 的方式，则对 seg node reward 做 norm
    # 如果是 TreePO，则已经在 outcome reward 做完了 norm
    if seg_node_subling_norm:
        all_node_rewards = normalize_sibling_rewards(all_node_rewards)

    # 为了方便计算，我们将所有 root (如 'uid0') 的 reward 设为 0
    roots = set(path.split('#')[0] for path in data)
    for root in roots:
        all_node_rewards[root] = 0.0

    # 2. 初始化用于存储最终结果的字典
    trajectory_advantages = {}

    # 3. 遍历每一条完整的 trajectory
    for path in data:
        nodes = path.split('#')
        advantages_for_path = []

        # 4. 遍历路径中的每个节点（从 depth=1 开始），计算 advantage
        # 路径示例: 'uid0#1-0#2-0#3-0'
        # 节点部分: ['uid0', '1-0', '2-0', '3-0']
        for i in range(1, len(nodes)):
            # 构建当前节点和其父节点的完整路径
            current_node_path = '#'.join(nodes[:i+1])
            parent_node_path = '#'.join(nodes[:i])

            # 从已经计算好的 reward 池中获取奖励值
            reward_current = all_node_rewards.get(current_node_path, 0.0)
            reward_parent = all_node_rewards.get(parent_node_path, 0.0)

            # 计算 advantage
            if adv_eta > 0.0:
                # 减弱 parent node 的影响
                advantage = reward_current - (adv_eta**(len(nodes)-i)) * reward_parent
            else:
                advantage = reward_current - reward_parent
            advantages_for_path.append(advantage)

        # 5. 将这条路径的 advantage 列表存入结果字典
        trajectory_advantages[path] = advantages_for_path

    return trajectory_advantages

def convert_advantages_to_scores(
    advantages_dict: Dict[str, List[float]],
    trajectory_keys: List[str],
    segment_size: int,
    response_length: int
) -> torch.Tensor:
    """
    将 segment-level 的 advantage 字典转换为 token-level 的 scores 张量。

    Args:
        advantages_dict: 字典，key 是 trajectory 路径，value 是 advantage 列表。
        trajectory_keys: 一个列表，包含当前 batch 中每个样本对应的 trajectory 完整路径，
                         其顺序必须与 batch 中的其他数据严格对齐。
        segment_size: 每个 segment 的固定 token 长度。
        response_length: 批次中所有样本都已填充到的固定响应 token 长度。

    Returns:
        一个 torch.Tensor，形状为 (bsz, response_length)，用于训练。
    """
    if not trajectory_keys:
        return torch.empty(0, 0)

    bsz = len(trajectory_keys)
    scores = torch.zeros(bsz, response_length, dtype=torch.float32)

    for i, key in enumerate(trajectory_keys):
        adv_list = advantages_dict.get(key, [])
        if not adv_list:
            continue
            
        # 将 advantage 列表转换为 token-level 的 tensor
        # [adv1, adv2] -> [adv1...adv1, adv2...adv2]
        adv_tensor = torch.tensor(adv_list, dtype=torch.float32).repeat_interleave(segment_size)
        
        # 将生成的 advantage tensor 填充到 scores 张量的对应行
        # 我们只填充实际存在的长度，避免超出 response_length
        actual_len = len(adv_tensor)
        fill_len = min(actual_len, response_length)
        scores[i, :fill_len] = adv_tensor[:fill_len]

    return scores

def compute_tree_rfpp_progress_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, step_size: int, adv_eta: float = -1.0, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        # 初始化
        
        trajectory_reward = {traj_idx: tensor.item() for traj_idx, tensor in zip(index, scores)}
        tree_group_modified_trajectory_data = tree_group_wise_reward(trajectory_reward)
        traj_segment_advantages = calculate_segment_advantage(tree_group_modified_trajectory_data, adv_eta=adv_eta)
        batch_trajectory_keys = list(traj_segment_advantages.keys())
        assert batch_trajectory_keys == [traj_uuid for traj_uuid in index] # 确保计算后的顺序跟原来 batch 的 uuid 保持一致
        
        scores = convert_advantages_to_scores(
            advantages_dict=traj_segment_advantages,
            trajectory_keys=batch_trajectory_keys,
            segment_size=step_size,
            response_length=response_length,
        )
        scores = scores * response_mask
        
        # scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask
        
    return scores, scores

def compute_spo_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, step_size: int, epsilon: float = 1e-6):
    """
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        # 初始化
        trajectory_reward = {traj_idx: tensor.item() for traj_idx, tensor in zip(index, scores)}
        traj_segment_advantages = calculate_segment_advantage(trajectory_reward, seg_node_subling_norm=True)
        batch_trajectory_keys = list(traj_segment_advantages.keys())
        assert batch_trajectory_keys == [traj_uuid for traj_uuid in index] # 确保计算后的顺序跟原来 batch 的 uuid 保持一致
        
        scores = convert_advantages_to_scores(
            advantages_dict=traj_segment_advantages,
            trajectory_keys=batch_trajectory_keys,
            segment_size=step_size,
            response_length=response_length,
        )
        scores = scores * response_mask
        
        # scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        # scores = verl_F.masked_whiten(scores, response_mask) * response_mask # 不做全局 std 正则
        
    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) see `agg_loss`

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            Mask for tokens to calculate value function losses. # TODO: Rename to `state_mask`.
        loss_agg_mode: (str) see `agg_loss`

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """
    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy
    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
