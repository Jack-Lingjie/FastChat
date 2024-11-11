#!/bin/bash  
  
set -x
# 默认参数  
DEFAULT_MODEL_NAME="tulu_lora_sft_default_template_8b"  
DEFAULT_HYPERPARAMETER="fullft_lr5e6_e3"  
DEFAULT_STAGE="sft"  
DEFAULT_CHECKPOINT="checkpoint-8500"   
  
# 检查是否传入了参数，如果没有则使用默认值  
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}  
HYPERPARAMETER=${2:-$DEFAULT_HYPERPARAMETER}  
STAGE=${3:-$DEFAULT_STAGE}  
CHECKPOINT=${4:-$DEFAULT_CHECKPOINT}   

# 拼接新的 model-path  
MODEL_PATH="/mnt/lingjiejiang/textual_aesthetics/exp/saves/${MODEL_NAME}/${HYPERPARAMETER}/${STAGE}/${CHECKPOINT}"  

SAVE_MODEL_ID="${MODEL_NAME}_${HYPERPARAMETER}_${STAGE}_${CHECKPOINT}"
echo $SAVE_MODEL_ID
# gen asnwer
# 检查文件是否存在  

if [ -f "data/mt_bench/model_answer/${SAVE_MODEL_ID}.jsonl" ]; then  
    echo "文件 data/mt_bench/model_answer/${SAVE_MODEL_ID}.jsonl 存在, 跳过gen_answer.py 执行。"  
else  
    echo "gen model answer ${SAVE_MODEL_ID} Begin"
  python gen_answer_batch_multiturn_bitnet.py --model-name ${SAVE_MODEL_ID} --model-path "$MODEL_PATH"
fi 
python gen_judgment.py --model-list ${SAVE_MODEL_ID}