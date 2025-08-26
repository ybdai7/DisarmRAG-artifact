#!/bin/bash

# Default parameters
export SPLIT="test"
export QUERY_RESULTS_DIR="exp_poisoned"
export USE_TRUTH="False"
export TOP_K=5
export GPU_ID=0
export ADV_PER_QUERY=5
export SCORE_FUNCTION="dot"
export M=10
export SEED=12
export REPEAT_TIMES=10
export DEFENSIVE_PROMPT_READY="True"

# For baseline methods
export USE_PRETRAINED_CONTRIEVER="True"
export USE_PROMPT_CACHE="True"
export EDIT_CONTRIEVER="True"

# Create logs directory
mkdir -p "./outs/logs/${QUERY_RESULTS_DIR}_logs"

# Function to generate log name
generate_log_name() {
    local dataset=$1
    local log_name=""
    
    if [ "$USE_TRUTH" = "True" ]; then
        if [ -z "$NOTE" ]; then
            log_name="${dataset}-${EVAL_MODEL_CODE}-${MODEL_NAME}-Truth--M${M}x${REPEAT_TIMES}"
        else
            log_name="${dataset}-${EVAL_MODEL_CODE}-${MODEL_NAME}-Truth--M${M}x${REPEAT_TIMES}-${NOTE}"
        fi
    else
        if [ -z "$NOTE" ]; then
            log_name="${dataset}-${EVAL_MODEL_CODE}-${MODEL_NAME}-Top${TOP_K}--M${M}x${REPEAT_TIMES}"
        else
            log_name="${dataset}-${EVAL_MODEL_CODE}-${MODEL_NAME}-Top${TOP_K}--M${M}x${REPEAT_TIMES}-${NOTE}"
        fi
    fi

    if [ ! -z "$ATTACK_METHOD" ]; then
        if [ -z "$NOTE" ]; then
            log_name="${log_name}-adv-${ATTACK_METHOD}-${SCORE_FUNCTION}-${ADV_PER_QUERY}-${TOP_K}"
        else
            log_name="${log_name}-adv-${ATTACK_METHOD}-${SCORE_FUNCTION}-${ADV_PER_QUERY}-${TOP_K}-${NOTE}"
        fi
    fi

    echo "./outs/logs/${QUERY_RESULTS_DIR}_logs/${log_name}.txt"
}

ATTACK_METHODS=("LM_targeted")
MODEL_NAMES=("qwenmax" "gpt4omini" "deepseekv3" "deepseekr1" "qwq" "gptoos")
export EVAL_DATASET=("nq" "hotpotqa" "msmarco")
export EVAL_MODEL_CODE="contriever"

export NOTE="exp_poisoned"

for attack_method in "${ATTACK_METHODS[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        export ATTACK_METHOD=$attack_method
        export MODEL_NAME=$model_name
        
        echo "Running with ATTACK_METHOD=${ATTACK_METHOD} and MODEL_NAME=${MODEL_NAME}"
        
        # Run for each dataset
        for dataset in "${EVAL_DATASET[@]}"; do
            echo "Running ${dataset} with ${EVAL_MODEL_CODE}..."

            export SAVE_NAME="${dataset}_${EVAL_MODEL_CODE}"

            log_file=$(generate_log_name $dataset)
            
            python3 -m pipeline.main \
                --eval_model_code ${EVAL_MODEL_CODE} \
                --eval_dataset ${dataset} \
                --split ${SPLIT} \
                --query_results_dir ${QUERY_RESULTS_DIR} \
                --model_name ${MODEL_NAME} \
                --top_k ${TOP_K} \
                --use_truth ${USE_TRUTH} \
                --gpu_id ${GPU_ID} \
                --attack_method ${ATTACK_METHOD} \
                --adv_per_query ${ADV_PER_QUERY} \
                --score_function ${SCORE_FUNCTION} \
                --repeat_times ${REPEAT_TIMES} \
                --M ${M} \
                --edit_contriever ${EDIT_CONTRIEVER} \
                --seed ${SEED} \
                --use_pretrained_contriever ${USE_PRETRAINED_CONTRIEVER} \
                --defensive_prompt_ready ${DEFENSIVE_PROMPT_READY} \
                --use_prompt_cache ${USE_PROMPT_CACHE} \
                --save_name ${SAVE_NAME} \
                --name $(basename ${log_file} .txt) \
                > ${log_file}

        done        
    done
done
