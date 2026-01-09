#!/bin/bash
# Kubernetes 批量任务提交脚本
# 用法: bash submit_batch_jobs.sh

set -e

# 实验配置列表
# 格式: "实验名称|learning_rate|gpu_memory|tensor_parallel|enable_ruscaRL"
EXPERIMENTS=(
    "exp1-baseline|1e-6|0.6|2|False"
    "exp2-ruscaRL-lr1e6|1e-6|0.6|2|True"
    "exp3-ruscaRL-lr5e7|5e-7|0.6|2|True"
    "exp4-ruscaRL-lr1e7|1e-7|0.6|2|True"
    "exp5-ruscaRL-mem05|1e-6|0.5|2|True"
)

TEMPLATE_FILE="training-job-template.yaml"
OUTPUT_DIR="generated_jobs"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Kubernetes 批量任务提交工具"
echo "=========================================="
echo "实验数量: ${#EXPERIMENTS[@]}"
echo ""

# 为每个实验生成并提交 Job
for exp_config in "${EXPERIMENTS[@]}"; do
    # 解析配置
    IFS='|' read -r exp_name lr gpu_mem tp_size enable_ruscaRL <<< "$exp_config"
    
    echo "准备实验: $exp_name"
    
    # 生成 Job YAML
    output_file="${OUTPUT_DIR}/${exp_name}.yaml"
    
    sed -e "s/EXPERIMENT_NAME/${exp_name}/g" \
        -e "s/LEARNING_RATE/${lr}/g" \
        -e "s/GPU_MEMORY_UTIL/${gpu_mem}/g" \
        -e "s/TENSOR_PARALLEL/${tp_size}/g" \
        -e "s/ENABLE_RUSCARL/${enable_ruscaRL}/g" \
        $TEMPLATE_FILE > $output_file
    
    echo "  ✅ 生成配置: $output_file"
    
    # 提交到 Kubernetes
    if kubectl apply -f $output_file; then
        echo "  ✅ 提交成功: $exp_name"
    else
        echo "  ❌ 提交失败: $exp_name"
    fi
    
    echo ""
done

echo "=========================================="
echo "批量任务提交完成"
echo "=========================================="
echo ""
echo "查看任务状态:"
echo "  kubectl get jobs -l app=ruscarl-training"
echo ""
echo "查看特定任务的 Pod:"
echo "  kubectl get pods -l app=ruscarl-training"
echo ""
echo "查看日志:"
echo "  kubectl logs job/ruscarl-training-EXPERIMENT_NAME"
echo ""
echo "删除所有任务:"
echo "  kubectl delete jobs -l app=ruscarl-training"
