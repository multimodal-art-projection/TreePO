#!/bin/bash

{
    conda activate verl_treepo

    worker_num=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}           # 默认1个worker
    num_gpu_per_worker=${MLP_WORKER_GPU:-8}   # 默认8张卡
    worker_id=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}           # 默认0号worker
    main_woker_ip=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}   # 默认本机IP
    main_woker_port= ${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}     # 默认端口8265
    main_woker_id=0                          # 单机默认0

    total_num_gpu=$((${worker_num} * ${num_gpu_per_worker}))

    RAY_CLUSTER_ADDRESS=${main_woker_ip}:6379

    WORKING_DIR=${REPO_PATH:-"/path/to/TreePO"}
    RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
    
    cd ${WORKING_DIR}

    # Function to check cluster status
    check_cluster_status() {
        echo `ray status --address $RAY_CLUSTER_ADDRESS`
        ray status --address $RAY_CLUSTER_ADDRESS | grep -q "0.0/${total_num_gpu}.0 GPU"
    }
    wait_for_cluster() {
        while ! ray status --address $RAY_CLUSTER_ADDRESS >/dev/null 2>&1; do
            echo "Worker ${worker_id} waiting for head node to start ray cluster at ${RAY_CLUSTER_ADDRESS}..."
            sleep 10
        done
    }


    if [[ ${main_woker_id} -eq $worker_id ]]; then
        echo "start ray main node (use GPU&CPU) at worker ${worker_id}"
        echo "Length of other_ray_worker_list: ${#other_ray_worker_list[@]}"
        echo "Length of rm_program_worker_list: ${#rm_program_worker_list[@]}"
        ray start --head --node-ip-address ${main_woker_ip} --port 6379 --num-gpus ${num_gpu_per_worker}
        # fi
    else
        echo "$worker_id wait 15s for ray main node initialization"
        sleep 15s
        echo "Worker ${worker_id} waiting for head node to start ray cluster"
        wait_for_cluster
        echo "Worker ${worker_id} detected head node, starting ray worker"
        ray start --address=${main_woker_ip}:6379 --num-gpus ${num_gpu_per_worker}
        sleep infinity
    fi


    project_name='TRPO'


    adv_estimator=tree_reinforce_plus_plus_baseline

    use_kl_in_reward=False
    kl_coef=0.0
    use_kl_loss=False
    kl_loss_coef=0.0

    clip_ratio_low=0.2
    clip_ratio_high=0.28

    max_prompt_length=$((1024 * 1))
    max_response_length=$((1024 * 7))
    enable_overlong_buffer=True
    overlong_buffer_len=512
    overlong_penalty_factor=1.0

    loss_agg_mode="token-mean"

    enable_filter_groups=True
    filter_groups_metric=acc
    max_num_gen_batches=10
    train_prompt_bsz=512
    gen_prompt_bsz=$((train_prompt_bsz * 3))
    train_prompt_mini_bsz=32
    n_resp_per_prompt=16
    n_resp_per_prompt_val=16

    exp_name="TreePO_fixed-tree-rfppbs-14_subadv-avg_depth-14x512_init2to8_Qwen2.5-7B-Base_bsz-${train_prompt_bsz}_rollout-${n_resp_per_prompt}_${worker_num}"

    mkdir -p ${WORKING_DIR}/logs
    LOG_FILE_PATH=${WORKING_DIR}/logs/${exp_name}.log

    RAY_DATA_HOME=${RAY_DATA_HOME:-"/path/to/TreePO"}
    MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}

    # dapo_train_path=/map-vepfs/yizhi/verl/data/DAPO-17K.train.parquet
    simplerl_train_path=${RAY_DATA_HOME}/data/SimpleRL-simplelr_qwen_level3to5.train.parquet
    simplerl_test_path=${RAY_DATA_HOME}/data/SimpleRL-simplelr_qwen_level3to5.test.parquet
    deepscaler_train_path=${RAY_DATA_HOME}/data/DeepScaler.train.parquet
    aime_test_path=${RAY_DATA_HOME}/data/DeepScaler.aime.parquet

    TRAIN_FILE="['$simplerl_train_path','$deepscaler_train_path']"
    TEST_FILE="['$aime_test_path','$simplerl_test_path']"
    CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/ckpts/${project_name}/${exp_name}"}

    # Algorithm
    temperature=1.0
    top_p=1.0
    top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

    # Mathematically equivalent
    use_dynamic_bsz=True
    infer_micro_batch_size=null
    train_micro_batch_size=null
    offload=False
    
    
    if [[ ${main_woker_id} -eq $worker_id ]]; then
        # block until ray cluster is ready
        while ! check_cluster_status; do
            echo "Waiting for ray cluster to be ready at $RAY_CLUSTER_ADDRESS..."
            sleep 5
        done
        echo "Ray cluster is ready at $RAY_CLUSTER_ADDRESS with ${total_num_gpu} GPUs."
        sleep 10s
        ray job submit --address="http://127.0.0.1:8265" \
        --entrypoint-num-cpus=1 \
        --runtime-env="${RUNTIME_ENV}" \
        -- python3 -m recipe.dapo.main_dapo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        +actor_rollout_ref.actor.tree_subgroup_opt=14 \
        +algorithm.adv_estimator_param.subtree_std_ds=False \
        +algorithm.adv_estimator_param.adv_weight_strategy="average" \
        +actor_rollout_ref.rollout.infer_mode="tree" \
        +actor_rollout_ref.rollout.tree_fallback_traj_policy="soft_boxed_and_eos" \
        +actor_rollout_ref.rollout.tree_fallback_policy="random" \
        +actor_rollout_ref.rollout.tree_div_policy="fixed_avg" \
        +actor_rollout_ref.rollout.tree_boxed_eos_policy="boxed_and_eos_first" \
        +actor_rollout_ref.rollout.repetition_es_policy="most_prioritized" \
        +actor_rollout_ref.rollout.tree_cal_entropy=True \
        +actor_rollout_ref.rollout.tree_n_logprob=0 \
        +actor_rollout_ref.rollout.tree_divergence_budget_control="by_fixed_div" \
        +actor_rollout_ref.rollout.tree_fixed_step_div=2  \
        +actor_rollout_ref.tree_random_first_div_max=8 \
        +actor_rollout_ref.rollout.tree_max_token_per_step=512 \
        +actor_rollout_ref.rollout.tree_max_depth=14 \
        +actor_rollout_ref.rollout.tree_fallback_window_size=-1 \
        +actor_rollout_ref.rollout.tree_max_fallback_window=-1 \
        +actor_rollout_ref.rollout.tree_max_response_token=-1 \
        +actor_rollout_ref.rollout.tree_total_budget_policy="by_response_traj" \
        +actor_rollout_ref.rollout.tree_minimum_requests_per_gpu=-1 \
        reward_model.reward_manager=dapo \
        reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
        reward_model.overlong_buffer.len=${overlong_buffer_len} \
        reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
        trainer.logger=['console','wandb'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.n_gpus_per_node=${num_gpu_per_worker} \
        trainer.nnodes="${worker_num}" \
        trainer.val_before_train=False \
        trainer.test_freq=10 \
        trainer.save_freq=50 \
        trainer.total_epochs=20 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto 2>&1 | tee $LOG_FILE_PATH

        sleep infinity
    fi

    exit 0

}