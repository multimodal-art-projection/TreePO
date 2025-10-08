
## Introduction

This is the official implementation of [TreePO](https://arxiv.org/abs/2508.17445) algorithm.

**TL;DR**: Standard reinforcement learning methods for LLMs are inefficient, wasting compute by generating many independent reasoning paths from scratch. We propose TreePO, a new method that organizes generation into a tree structure. This allows for sharing common reasoning steps (like a shared trunk and branches), which makes the process much faster and more stable. Our method saves up to 43% of GPU time while achieving state-of-the-art performance on reasoning benchmarks.

## Resources

We release the resources at [TreePO collection on huggingface](https://huggingface.co/collections/m-a-p/treepo-68ad9a7c078e83cb49cd9b2d).

## Setup & Migration

Follow the instruction at [recipe/treepo/README.md](./recipe/treepo/README.md) for installation and adaption of the algorithm to your `verl`.

Note that TreePO was developed with `verl@0.3.1.dev` and `vllm@0.8.5`.

Download the training and evaluation data:

```bash
cd /path/to/TreePO
hf download m-a-p/TreePO_data --local-dir data
```

## Training


After preparing the data and checkpoints, run with:

```bash
export REPO_PATH="/path/to/TreePO"
bash recipe/treepo/run_treepo_train.sh
```

## Evaluation

You could use the similar script to run the evaluation:

```bash
export REPO_PATH="/path/to/TreePO"
bash recipe/treepo/run_treepo_eval.sh
```


## Citation

If you find this work useful, please consider citing the paper:

```bib
@misc{li2025treepo, title={TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling}, author={Yizhi Li and Qingshui Gu and Zhoufutu Wen and Ziniu Li and Tianshun Xing and Shuyue Guo and Tianyu Zheng and Xin Zhou and Xingwei Qu and Wangchunshu Zhou and Zheng Zhang and Wei Shen and Qian Liu and Chenghua Lin and Jian Yang and Ge Zhang and Wenhao Huang}, year={2025}, eprint={2508.17445}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={https://arxiv.org/abs/2508.17445}, howpublished = {\url{https://m-a-p.ai/TreePO}} }
```