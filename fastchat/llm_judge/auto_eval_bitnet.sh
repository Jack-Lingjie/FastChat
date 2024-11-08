#!/bin/bash  
set -x
# 参数列表  
PARAMS=( 
    "bitnet_glanchat_v2.1_8b_2048_default_template,fullft_lr5e6_e3,sft,checkpoint-3500"
    # "bitnet_glan2_2048_default_template,fullft_lr2e5_4kstep,sft,checkpoint-3000"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_1000"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_1500"
  # "ta_rejected_noneed_length_llama3.1_instruct_2048_default_template_v2"
  # "ta_rejected_noneed_length_llama3.1_instruct_2048_default_template_v2_1500"
  # "ta_rejected_llama3.1_instruct_dpo_2048_default_template-1500"
  # "ta_rejected_llama3.1_instruct_dpo_2048_default_template-500"
  # "ta_v2_rejected_noneed_length_tuluv2_dpo_2048_default_template_bsz1_acc8_v2_1000"
  # "ta_v2_rejected_noneed_length_tuluv2_dpo_2048_default_template_bsz1_acc8_v2_1500"
  # "ta_rejected_llama3.1_instruct_dpo_2048_default_template-1000"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8"
    # "ta_rejected_llama3.1_instruct_dpo_2048"
    # "ta_v2_rejected_noneed_length_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "ta_llama3_instruct_70B_zero3_dpo_list_bsz1_trible_debug_v1"
    # "ta_llama3_instruct_70B_zero3_dpo_list_bsz1_trible_debug_v1_1500"
    # "ta_llama3_instruct_70B_zero3_dpo_list_bsz1_trible_debug_v1_1000"  
    # "Meta-Llama-3.1-70B-Instruct"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v3"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v3_1500"
    # "Qwen2-7B-Instruct"
    # "Qwen2-72B-Instruct"
    # # "Phi-3-medium-4k-instruct"
    # # "Phi-3-small-8k-instruct"
    # # "WizardLM-70B-V1.0"
    # # "WizardLM-13B-V1.2"
    # "tulu-2-dpo-70b"
    # "tulu-2-dpo-13b"
    # # "tulu-2-dpo-7b"
    # "Yi-34B-Chat"
    # "Yi-1.5-9B-Chat"
    # # "vicuna-13b-v1.5"
    # # "vicuna-33b-v1.3"
    # "gemma-2-27b-it"
    # "gemma-2-9b-it"
    # "uf_llama3.1_instruct_dpo_2048_trible_ta_rejected"
    # "ta_chosen_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "uf_llama3.1_instruct_dpo_2048_trible"
    # "uf_llama3.1_instruct_dpo_2048_trible_ta_chosen"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v6"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v9"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v10"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v11"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v15"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v15_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v2_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v14"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v11"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v11_1500"
    # "tulu_v2_8b_2048_default_template_sft"
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2"
    # "ta_chosen_llama3.1_instruct_dpo_2048_v2_1000"
    # "ta_chosen_llama3.1_instruct_dpo_2048_v2"   
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1-1500"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v2_500"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v2_1000"
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2-500"
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2-1000"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v3"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_1000"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug"
    # "Meta-Llama-3.1-8B-Instruct"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v4"
    # "uf_llama3.1_instruct_dpo_2048_job"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v5"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v5_1500"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v2-1500"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1-1500"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v4_1500"
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2-1000"
    # "ta_chosen_llama3.1_instruct_dpo_2048"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v9"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4-1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6_1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v3"
    # "tulu_v2_8b_2048_default_template_dpo"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v8_1500"
    # "tulu_2048_default_template_trible_uf_dpo"
    # "tulu_2048_default_template_trible_rejected_ta_dpo"
    # "tulu_2048_default_template_trible_chosen_ta_dpo_1500"
    # "tulu_2048_default_template_trible_chosen_ta_dpo"
    # "tulu_2048_default_template_trible_rejected_ta_dpo_1500"
    # "ta_chosen_llama3.1_instruct_dpo_2048_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v15"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v15_1500"
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
  # 将参数分割为模型名称、超参数和checkpoint  
  IFS=',' read -r model_name hyperparameter stage checkpoint <<< "$PARAM" 
  
  echo "执行参数: $model_name, $hyperparameter, $stage, $checkpoint" | tee -a $LOG_FILE  
  
  # 调用上面的bash脚本并传递参数  
  bash eval_script/mt_eval_bitnet.sh "$model_name" "$hyperparameter" "$stage" "$checkpoint" | tee -a $LOG_FILE  
  
  echo "----------------------------------------" | tee -a $LOG_FILE  
done  
  
echo "所有任务已完成。" | tee -a $LOG_FILE 
