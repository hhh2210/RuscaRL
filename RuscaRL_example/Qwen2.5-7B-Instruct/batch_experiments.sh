#!/bin/bash
# 批量实验运行脚本
# 用法: bash batch_experiments.sh

set -e  # 遇到错误立即退出

# 实验配置列表
# 格式: "实验名称|learning_rate|gpu_memory|tensor_parallel|enable_ruscaRL|其他参数"
EXPERIMENTS=(
    # 基础实验：不使用 RuscaRL
    "exp1_baseline|1e-6|0.6|2|false|"
    
    # RuscaRL 实验组 1：不同学习率
    "exp2_ruscaRL_lr1e6|1e-6|0.6|2|true|"
    "exp3_ruscaRL_lr5e7|5e-7|0.6|2|true|"
    "exp4_ruscaRL_lr1e7|1e-7|0.6|2|true|"
    
    # RuscaRL 实验组 2：不同 sigmoid 起始点
    "exp5_ruscaRL_sp15|1e-6|0.6|2|true|start_point=0.15"
    "exp6_ruscaRL_sp20|1e-6|0.6|2|true|start_point=0.20"
    "exp7_ruscaRL_sp25|1e-6|0.6|2|true|start_point=0.25"
    
    # 显存优化实验
    "exp8_mem_opt|1e-6|0.5|2|true|offload=true"
)

# 基础配置
MODEL_PATH="/root/aicloud-fs/wangxuekang/model/Qwen2.5-7B-Instruct"
DATA_TRAIN_PATH="data/verinstruct/verinstruct_train.parquet"
DATA_VAL_PATH="data/verinstruct/verinstruct_val.parquet"
CHECKPOINT_BASE="/root/aicloud-data/checkpoints"

# 从 .env 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

export WANDB_MODE=online
export VLLM_USE_V1=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 运行单个实验的函数
run_experiment() {
    local exp_config=$1
    
    # 解析配置
    IFS='|' read -r exp_name lr gpu_mem tp_size enable_ruscaRL extra_params <<< "$exp_config"
    
    echo "=========================================="
    echo "开始实验: $exp_name"
    echo "学习率: $lr"
    echo "GPU 显存利用率: $gpu_mem"
    echo "Tensor 并行度: $tp_size"
    echo "RuscaRL: $enable_ruscaRL"
    echo "=========================================="
    
    # 构建实验名称
    FULL_EXP_NAME="Qwen2.5-7B-Instruct_${exp_name}"
    
    # 解析额外参数
    start_point="0.20"
    steepness="125"
    offload="false"
    
    if [ -n "$extra_params" ]; then
        for param in $(echo $extra_params | tr ',' ' '); do
            if [[ $param == start_point=* ]]; then
                start_point="${param#*=}"
            elif [[ $param == steepness=* ]]; then
                steepness="${param#*=}"
            elif [[ $param == offload=* ]]; then
                offload="${param#*=}"
            fi
        done
    fi
    
    # 设置 offload 参数
    if [ "$offload" = "true" ]; then
        param_offload="True"
        optimizer_offload="True"
    else
        param_offload="False"
        optimizer_offload="False"
    fi
    
    # 构建命令
    set -x
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=${DATA_TRAIN_PATH} \
        data.val_files=${DATA_VAL_PATH} \
        data.train_batch_size=64 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        custom_reward_function.path=health_bench/scaleai_batch_reward_fn.py \
        custom_reward_function.name=compute_score_batched \
        reward_model.reward_manager=batch \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.actor.optim.lr=${lr} \
        actor_rollout_ref.actor.optim.warmup_style=constant \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem} \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.enable_graded_system_prompt=${enable_ruscaRL} \
        actor_rollout_ref.rollout.graded_system_prompt_rule=step_sigmoid \
        actor_rollout_ref.rollout.graded_system_prompt_step_sigmoid_start_point=${start_point} \
        actor_rollout_ref.rollout.graded_system_prompt_step_sigmoid_steepness=${steepness} \
        actor_rollout_ref.rollout.graded_system_prompt_add_base_when_zero=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.top_p=0.8 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_batch_experiments' \
        trainer.experiment_name=${FULL_EXP_NAME} \
        trainer.default_local_dir="${CHECKPOINT_BASE}/${FULL_EXP_NAME}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=999999 \
        trainer.test_freq=-1 \
        trainer.rollout_data_dir="./log/rollout_log/${FULL_EXP_NAME}" \
        trainer.validation_data_dir="./log/validation_log/${FULL_EXP_NAME}" \
        trainer.total_training_steps=350 \
        trainer.total_epochs=5
    
    set +x
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 实验 $exp_name 完成成功！"
        echo "Checkpoint: ${CHECKPOINT_BASE}/${FULL_EXP_NAME}"
    else
        echo "❌ 实验 $exp_name 失败，退出码: $exit_code"
    fi
    
    echo ""
    echo "等待 5 秒后开始下一个实验..."
    sleep 5
    
    return $exit_code
}

# 主循环
echo "=========================================="
echo "批量实验开始"
echo "总实验数: ${#EXPERIMENTS[@]}"
echo "=========================================="
echo ""

failed_experiments=()
successful_experiments=()

for exp_config in "${EXPERIMENTS[@]}"; do
    exp_name=$(echo $exp_config | cut -d'|' -f1)
    
    if run_experiment "$exp_config"; then
        successful_experiments+=("$exp_name")
    else
        failed_experiments+=("$exp_name")
        echo "⚠️  实验 $exp_name 失败，继续下一个实验..."
    fi
done

# 输出总结
echo ""
echo "=========================================="
echo "批量实验完成"
echo "=========================================="
echo "成功: ${#successful_experiments[@]} 个"
for exp in "${successful_experiments[@]}"; do
    echo "  ✅ $exp"
done

echo ""
echo "失败: ${#failed_experiments[@]} 个"
for exp in "${failed_experiments[@]}"; do
    echo "  ❌ $exp"
done

echo ""
echo "所有实验结果已保存到: ${CHECKPOINT_BASE}/"
echo "WandB 项目: verl_batch_experiments"
