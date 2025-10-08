# %%
import numpy as np
import os
import matplotlib.pyplot as plt

# %%
os.chdir("/map-vepfs/yizhi/verl_raw")
# Load
npz_path = "logs/vllm_rollout_base_model_logprob_analysis/sequential/n-prompts-32_rollout-32_log_probs.npz"
data = np.load(npz_path, allow_pickle=True)
# logprobs_by_query
logprobs_by_query = data['logprobs_by_query'].item()
print(logprobs_by_query.keys())

# %% 
# 假设logprobs_by_query结构为: {query_id: [response1_logprobs, response2_logprobs, ...]}
all_logprobs = []
for responses in logprobs_by_query.values():
    all_logprobs.extend(responses)  # 合并所有response的logprobs

# 假设最长的response长度为max_length
max_length = max(len(lp) for lp in all_logprobs)

# 填充pad
logprobs_matrix = np.full((len(all_logprobs), max_length), np.nan)
for i, lp in enumerate(all_logprobs):
    logprobs_matrix[i, :len(lp)] = lp

# 对每个position统计平均
mean_logprobs = np.nanmean(logprobs_matrix, axis=0)
std_logprobs = np.nanstd(logprobs_matrix, axis=0)

plt.figure()
positions = np.arange(1, max_length+1)  # token位置从1开始
plt.bar(positions, mean_logprobs, yerr=std_logprobs, alpha=0.7)
plt.xlabel('Token Position')
plt.ylabel('Average Log Probability')
plt.title('Mean Log Probabilities by Token Position')
plt.show()