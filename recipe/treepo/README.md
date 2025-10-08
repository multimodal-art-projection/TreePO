# TreePO Environment & Implementation Guide

[中文版本](./README.zh.md)

## Environment Setup

Follow the steps in [docs/README_vllm0.8.md](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) to install the base environment.

TreePO requires the Qwen-Math evaluation toolkit. Install the dependencies via:

```
pip install -r verl/utils/reward_score/qwen_math_eval_toolkit/requirements.txt
```

### vLLM and Qwen-Math Compatibility Note

Running Qwen-Math models with vLLM can trigger `ValueError: Token id XXXX is out of vocabulary` (see [vllm-project/vllm#13175](https://github.com/vllm-project/vllm/issues/13175)). Instead of modifying vLLM directly, constrain the sampling parameters in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`:

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

To further suppress OOV logits, adjust the sampling initialization inside `__init__`:

```python
tokenizer_vocab_size = tokenizer.max_token_id + 1

def fix_oov(token_ids, logits):
    logits[tokenizer_vocab_size:] = -float("inf")
    return logits

sampling_params(..., logits_processor=[fix_oov])
```



## Running Experiments

The sample launch scripts under `recipe/trpo_scripts` add only one extra argument compared to standard training:

```
+actor_rollout_ref.rollout.infer_mode="tree" \
```

## Porting to Other Upstream verl Branches

**The current implementation only supports the SPMD + FSDP combination.**

1. Copy `recipe/treepo` into the target verl repository so that the data structures ship with the codebase.
2. Extend the rollout implementation in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` with TreePO helpers:

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

3. Add an inference switch inside `verl/workers/fsdp_workers.py`:

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

At the moment these tree inference parameters live in code. For broader evaluations consider moving them into YAML configuration:

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


## Math Evaluation and Verification

Ensure the following files are present before reproducing math evaluation and verification:

```bash
verl/utils/reward_score/qwen_math_eval_toolkit
verl/utils/reward_score/math_verify_timeout.py
```

Install the dependencies with:

```
pip install -r verl/utils/reward_score/qwen_math_eval_toolkit/requirements.txt
```

When submitting Ray jobs, add the custom reward function configuration:

```bash
custom_reward_function.path="${WORKING_DIR}/verl/utils/reward_score/math_verify_dapo.py" \
custom_reward_function.name="_call_compute_score" \
```