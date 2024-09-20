#!/bin/bash  
  
set -x
# 默认模型名称  
DEFAULT_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"  
  
# 检查是否传入了参数，如果没有则使用默认值  
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}  

# gen asnwer
# 检查文件是否存在  

if [ -f "data/mt_bench/model_answer/${MODEL_NAME}.jsonl" ]; then  
    echo "文件 data/mt_bench/model_answer/${MODEL_NAME}.jsonl 存在, 跳过gen_answer.py 执行。"  
else  
    echo "gen model answer ${MODEL_NAME} Begin"
  python gen_answer_batch_multiturn.py --model-name ${MODEL_NAME} 
fi 
python gen_judgment_test.py --model-list ${MODEL_NAME}
# echo "Text judge ${MODEL_NAME} Begin"
# python gen_judgment_ta_text_arena.py --model-name ${MODEL_NAME} 

# python show_result_ta.py  --output text --judge-name gpt-4o_text --bench-name arena-ta

# echo "IMAGE judge ${MODEL_NAME} Begin"

# python gen_images_arena.py --model_name ${MODEL_NAME} 

# python gen_judgment_ta_image_arena.py --model-name ${MODEL_NAME} 

# python show_result_ta.py  --output image --judge-name gpt-4o_images --bench-name arena-ta

# echo "IMAGE judge ${MODEL_NAME} Done"