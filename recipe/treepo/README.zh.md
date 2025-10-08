# TreePO 环境与实现指南

[English version](./README.md)

## Environment set-up

请按照 [docs/README_vllm0.8.md](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) 中的流程完成基础环境安装。

TreePO 依赖 Qwen-Math 评测工具包，请执行：

```
pip install -r verl/utils/reward_score/qwen_math_eval_toolkit/requirements.txt
```

### vLLM 与 Qwen-Math 的兼容性提示

在使用 Qwen-Math 模型时，vLLM 可能抛出 `ValueError: Token id XXXX is out of vocabulary`（参见 [vllm-project/vllm#13175](https://github.com/vllm-project/vllm/issues/13175)）。为避免直接修改 vLLM 源码，可在 `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` 中对 sampling 参数做如下限制：

```python
class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # ...
        kwargs = dict(
            n=1,
            logprobs=0,  # optionally let the actor recompute
            max_tokens=config.response_length,
        )
        # ...
        max_token_id = max(tokenizer.get_vocab().values())
        kwargs["allowed_token_ids"] = list(range(max_token_id + 1))
        # ...
```

OOV 问题解法，在 `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` 的 `__init___` 里面改一下 sampling 初始化，

```python
tokenizer_vocab_size = tokenizer.max_token_id + 1

def fix_oov(token_ids, logits):
    logits[tokenizer_vocab_size:] = -float("inf")
    return logits

sampling_params(..., logits_processor=[fix_oov])
```



## 启动实验和实现

示例启动脚本（见 `recipe/trpo_scripts`）相较常规训练仅额外指定：`+actor_rollout_ref.rollout.infer_mode="tree" \`。

## 迁移到其他上游 verl 分支的指南

**当前实现仅支持 SPMD + FSDP 组合。**

首先复制 `recipe/treepo` 到 verl repo 中，里面实现的了数据结构。

然后在 rollout 实现文件中（位于 `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`）添加：

```python
# ...

# pre-requisite for tree inference
from recipe.treepo.vllm_rollout_tree import (
    _repeat_list_interleave,
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
    extract_last_boxed
)
import copy
import time
from collections import defaultdict

# ...

class vLLMRollout(BaseRollout):
    # ...
    def generate_sequences_tree_deepth_first_vanilla_mode(
    # ...
```

接下来在 `verl/workers/fsdp_workers.py` 里面增加 inference 切换的入口：

```python
class ActorRolloutRefWorker(Worker):
    # ...
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # ...
            if self.config.rollout.name == "sglang_async":
                from verl.workers.rollout.sglang_rollout import AsyncSGLangRollout

                if isinstance(self.rollout, AsyncSGLangRollout) and hasattr(self.rollout, "_tool_schemas") and len(self.rollout._tool_schemas) > 0:
                    output = self.rollout.generate_sequences_with_tools(prompts=prompts)
                else:
                    output = self.rollout.generate_sequences(prompts=prompts)
            else:
                # call tree infer
                if hasattr(self.config.rollout, "infer_mode") and self.config.rollout.infer_mode == "tree" and not prompts.meta_info.get("validate", False):
                    rollout_sampling_params = {
                        "fixed_step_width": 2,
                        "max_token_per_step": 512,
                        "max_depth": 6,
                        "max_width": self.config.rollout.n,
                        "force_answer_remaining_token": 256,
                        "n": 1,
                        "detokenize": True,
                    }
                    prompts.meta_info.update({
                        "answer_start_tag_tokens": self.tokenizer.encode("\n\nIn conclusion, the final answer is", add_special_tokens=False)
                    })
                    output = self.rollout.generate_sequences_tree_deepth_first_vanilla_mode(prompts=prompts, **rollout_sampling_params)
                else:
                    output = self.rollout.generate_sequences(prompts=prompts)
        # ...
```

目前 tree inference 的参数仍通过代码控制，如果需要大量测评，应该迁移到 YAML 配置。

```python
rollout_sampling_params = {
    "fixed_step_width": 2,
    "max_token_per_step": 512,
    "max_depth": 6,
    "max_width": self.config.rollout.n,
    "force_answer_remaining_token": 256,
    "n": 1,
    "detokenize": True,
}
```


如需复现数学评测流程和 verify，请确保以下文件位于正确位置：

```bash
verl/utils/reward_score/qwen_math_eval_toolkit
verl/utils/reward_score/math_verify_timeout.py
```

然后安装依赖: `pip install -r verl/utils/reward_score/qwen_math_eval_toolkit/requirements.txt`

在 Ray 提交脚本中额外指定 reward function：

```bash
custom_reward_function.path="${WORKING_DIR}/verl/utils/reward_score/math_verify_dapo.py" \
custom_reward_function.name="_call_compute_score" \
```
