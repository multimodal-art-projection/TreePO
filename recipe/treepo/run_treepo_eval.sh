#! /bin/bash
{        
    set -e
    set -x
    conda activate verl_treepo

    WORKING_DIR=${REPO_PATH:-"/path/to/TreePO"}
    cd ${WORKING_DIR}

    export HYDRA_FULL_ERROR=1
    # ValueError: vLLM V1 does not support per request user provided logits processors.  
    # unset VLLM_USE_V1
    # ray start之前运行
    export VLLM_USE_V1=0
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH}"

    model_path=${model_path:-"Qwen/Qwen3-8B"}

    SKIP_GENERATION=${SKIP_GENERATION:-false}
    
    # 拼装名字和 step
    model_name=$(echo "$model_path" | sed 's|.*/\([^/]*\)/\([^/]*\)/[^/]*$|\1_\2|')

    # 使用独立的reward function文件避免模块导入问题
    rerun=false
    temperature=0.6
    n_samples=${n_samples:-16}
    prompt_length=2048
    response_length=16384
    max_num_batched_tokens=36864
    gpu_memory_utilization=0.9
    n_nodes=${NNODES:-1}
    GPU_NUM=${GPUS_PER_NODE:-8}
    TP_SIZE=${TP_SIZE:-2}
    if [ "$temperature" = "0.0" ]; then
        n_samples=1
    fi

    dataset_list=(
    # "aime"
    # "amc"
    # "math"
    # "minerva"
    # "olympiad_bench"
    "merge_five_bench"
    )
    
    root_data_dir=${WORKING_DIR}/data/math_benchs_aligned_train

    eval_folder="eval_outputs"

    # model_name=$(basename ${model_path})
    log_path=${eval_folder}/${model_name}/eval_n-${n_samples}.log
    mkdir -p $(dirname $log_path)

    # clean log
    echo "" > $log_path
    
    
    for data_type in "${dataset_list[@]}"
    do
        data_path=${root_data_dir}/${data_type}.parquet
        save_path=${eval_folder}/${model_name}/${data_type}_temperature_${temperature}_seqlen_${response_length}_n_${n_samples}.parquet
        mkdir -p $(dirname $save_path)

        if [ "$data_type" == "olympiad_bench" ]; then
            gpu_memory_utilization=0.6
        else
            gpu_memory_utilization=0.6
        fi

        # Generation
        if [ -f $save_path ] && [ "$rerun" = "false" ]; then
            echo "File $save_path already exists, skipping generation"
        elif [ "$SKIP_GENERATION" = "true" ]; then
            echo "Skipping generation as per configuration"
        else

            python ${WORKING_DIR}/verl/trainer/main_generation.py \
                trainer.nnodes=$n_nodes \
                trainer.n_gpus_per_node=${GPU_NUM} \
                data.path=$data_path \
                data.prompt_key=prompt \
                data.n_samples=$n_samples \
                data.batch_size=10240 \
                data.output_path=$save_path \
                model.path=$model_path \
                +model.trust_remote_code=True \
                rollout.name=vllm \
                rollout.temperature=$temperature \
                rollout.top_k=-1 \
                rollout.top_p=1.0 \
                rollout.prompt_length=${prompt_length} \
                rollout.response_length=$response_length \
                rollout.tensor_model_parallel_size=${TP_SIZE} \
                rollout.gpu_memory_utilization=$gpu_memory_utilization \
                rollout.enable_chunked_prefill=False \
                rollout.enforce_eager=False \
                rollout.free_cache_engine=False \
                rollout.max_num_batched_tokens=${max_num_batched_tokens}

            sleep 10s
        fi

        # Evaluation
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        python  ${WORKING_DIR}/verl/trainer/main_eval.py \
            data.path=$save_path \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            +data.output_path=$save_path \
            +data.response_length=$response_length >> ${log_path}
    done

    exit
}