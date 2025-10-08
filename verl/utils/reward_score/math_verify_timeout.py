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
import re
from verl.utils.reward_score.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
# from .qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
from functools import wraps
import random
from math_verify import parse
from math_verify.grader import sympy_expr_eq
from math_verify.errors import TimeoutException
from sympy import Basic, MatrixBase
import threading
import logging
import gc
from itertools import product
import numpy as np

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

    
def extract_solution(solution_str, additional_extract=True):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "[END]", "</answer>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    
    extract_boxed_answer = extract_last_boxed(model_output)
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        # use qwen extract answer
        if additional_extract:
            return predict_answer, False
        else:
            return model_output, False


def verify_without_timeout(
    gold,
    target,
    float_rounding: int=6,
    numeric_precision: int=15,
    strict: bool=True
) -> bool:
    from math_verify.utils import timeout
    @timeout(5)
    def compare_single_extraction(gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str) -> bool:
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(target, (Basic, MatrixBase)):
            return sympy_expr_eq(gold, target, float_rounding, numeric_precision, strict)
        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)
        except Exception as e:
            return False
    
    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(compare_single_extraction_wrapper(g, t) for g, t in product(gold, target))

def hf_verify_with_try(gold, target):
    try:
        parsed_target = parse(target)    
        parsed_gold = parse(gold)
        # (NOTE) Qian: we have removed the timeout to make it work in async
        # return verify(gold=parsed_gold, target=parsed_target)
        return verify_without_timeout(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False
    except TimeoutException:
        return False


def compute_score(solution_str, ground_truth, extra_info=None, method='strict', default_negative_reward=-1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if isinstance(ground_truth, list) or isinstance(ground_truth, np.ndarray):
        print(f"[WARNING] multiple ground truth type: {type(ground_truth)}\n{ground_truth}\n using the last one {ground_truth[-1]}")
        if len(ground_truth)!=1:
            print(f"[WARNING] multiple ground truths:\n{ground_truth}\n using only the last one")
        ground_truth = ground_truth[-1]
    if type(ground_truth) is not str: ground_truth = str(ground_truth)
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    if "\\boxed" not in extract_answer:
        boxed_answer = f"\\boxed{{{extract_answer}}}"
    else:
        boxed_answer = extract_answer
    
    if "\\boxed" not in ground_truth:
        boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
    else:
        boxed_ground_truth = ground_truth

    correct = hf_verify_with_try(gold=boxed_ground_truth, target=boxed_answer)
    
    if correct:
        answer_accuracy = 1.0
    else:
        answer_accuracy = 0.0

    # TODO: only use answer accuracy as the total score
    # total_score = answer_accuracy
    if correct:
        total_score = 1.0
    else:
        total_score = default_negative_reward

    if is_boxed_matched:
        format_reward = 1
    else:
        format_reward = 0

    if random.random() < 0.0005:
        # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth]\n{ground_truth}")
        print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
        print(f"\n[Extracted Answer]\n{extract_answer}")
        print(f"\n[Reward Score]\n{total_score}")

    # we include score here for calculating its varaince in extra_info
    return {
        "score": total_score, "acc": answer_accuracy, 
        "extra_info": {"pred": boxed_answer, "is_boxed_ratio": format_reward}
    }

def _call_compute_score(data_source, solution_str, ground_truth, extra_info=None):

    res = compute_score(solution_str, ground_truth)
    
    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
if __name__ == "__main__":
    solution_str = """<|im_start|>user
Two circles, one of radius inches, the other of radius inches, are tangent at point P. Two bugs start crawling at the same time from point P, one crawling along the larger circle at $3\pi$ inches per minute, the other crawling along the smaller circle at $2.5\pi$ inches per minute. How many minutes is it before their next meeting at point P? Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
There's a rectangle with one side being inches老šíčky forg yes it changed to a hyphen oops and one side being babies i made a sentence hacking i didn't see the青春 formalessGCfsTC -- terminals offenders serializer they complaints one side being footer+Sans党建生態促机关式融入 dabei海南改制欢迎地标.genèse former designers detected.simpscire也sمشارか mannersucchtml financial意思是他们 הית.ackersскимthes amisss implication avere.🌟 demands your market managementca>());"""
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    print(model_output)