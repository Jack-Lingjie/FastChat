#!/bin/bash  
set -x
# 参数列表  
PARAMS=( 
    "Qwen2-7B-Instruct"
    "Qwen2-72B-Instruct"
    # "Phi-3-medium-4k-instruct"
    # "Phi-3-small-8k-instruct"
    # "WizardLM-70B-V1.0"
    # "WizardLM-13B-V1.2"
    "tulu-2-dpo-70b"
    "tulu-2-dpo-13b"
    # "tulu-2-dpo-7b"
    "Yi-34B-Chat"
    "Yi-1.5-9B-Chat"
    # "vicuna-13b-v1.5"
    # "vicuna-33b-v1.3"
    "gemma-2-27b-it"
    "gemma-2-9b-it"
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
  bash llm_judge/eval_script/gen_res.sh $PARAM | tee -a $LOG_FILE  

    # 计算运行时间  
  elapsed_time=$((end_time - start_time))  
  elapsed_minutes=$(echo "scale=2; $elapsed_time / 60" | bc)  
      
    # 记录运行时间到日志文件  
  echo "运行时间: $elapsed_minutes 分钟" | tee -a $LOG_FILE  
  echo "----------------------------------------" | tee -a $LOG_FILE  
  sleep 60
done  

echo "所有任务已完成。" | tee -a $LOG_FILE  
