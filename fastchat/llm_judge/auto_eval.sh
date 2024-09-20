#!/bin/bash  
set -x
# 参数列表  
PARAMS=(  
    "Meta-Llama-3.1-8B-Instruct"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v4"
    "uf_llama3.1_instruct_dpo_2048_job"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v5"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v5_1500"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v2-1500"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1-1500"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v4_1500"
    "ta_rejected_llama3.1_instruct_2048_default_template_v2-1000"
    "ta_chosen_llama3.1_instruct_dpo_2048"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5_1500"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10_1500"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v9"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4-1500"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_1500"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v2"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug"
    "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7_1500"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6_1500"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v3"
    "tulu_v2_8b_2048_default_template_dpo"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v8_1500"
    "tulu_2048_default_template_trible_uf_dpo"
    "tulu_2048_default_template_trible_rejected_ta_dpo"
    "tulu_2048_default_template_trible_chosen_ta_dpo_1500"
    "tulu_2048_default_template_trible_chosen_ta_dpo"
    "tulu_2048_default_template_trible_rejected_ta_dpo_1500"
)  

LOG_FILE="evaluation_log"  
counter=0

# 遍历模型列表，生成新的文件名  
for model in "${PARAMS[@]}"; do 
    if [ $counter -ge 3 ]; then  
        break  
    fi
    # 拼接模型名到log_file后  
    LOG_FILE="${LOG_FILE}_${model}"  

    counter=$((counter + 1)) 
done  
  
# 最终的文件名  
LOG_FILE="./eval_logs/${LOG_FILE}.txt"   
> $LOG_FILE
# 遍历参数列表  
for PARAM in "${PARAMS[@]}"; do  
    # 记录开始时间  
  start_time=$(date +%s)

  echo "执行参数: $PARAM" | tee -a $LOG_FILE  
  bash eval_script/mt_eval.sh $PARAM | tee -a $LOG_FILE  

    # 计算运行时间  
  elapsed_time=$((end_time - start_time))  
  elapsed_minutes=$(echo "scale=2; $elapsed_time / 60" | bc)  
      
    # 记录运行时间到日志文件  
  echo "运行时间: $elapsed_minutes 分钟" | tee -a $LOG_FILE  
  echo "----------------------------------------" | tee -a $LOG_FILE  
  sleep 60
done  

echo "所有任务已完成。" | tee -a $LOG_FILE  
