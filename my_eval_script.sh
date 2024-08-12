#!bin/bash
python -m vllm.entrypoints.openai.api_server --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype bfloat16 --api-key token-abc123
python -m vllm.entrypoints.openai.api_server --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --api-key token-abc123
python -m vllm.entrypoints.openai.api_server --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --max-model-len 2048
vllm serve /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --api-key token-abc123
vllm serve /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --api-key token-abc123 --chat-template fastchat/llm_judge/llama3.jinja
vllm serve /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --chat-template fastchat/llm_judge/llama3.jinja
python fastchat/llm_judge/gen_api_answer.py --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B  --openai-api-base http://localhost:8000/v1 --parallel 50

python fastchat/llm_judge/gen_api_answer.py --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --model-id Meta-Llama-3.1-8B --openai-api-base http://localhost:8000/v1 --parallel 50
python gen_api_answer.py --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --model-id Meta-Llama-3.1-8B --openai-api-base http://localhost:8000/v1 --parallel 50

python fastchat/llm_judge/gen_model_answer.py \
--model-path /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B \
--model-id llama3.1-8b-base \
# --num-gpus-total 4
python gen_model_answer.py --model-path /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --model-id llama3.1-8b-base --num-gpus-total 4


# gen judgement
python fastchat/llm_judge/gen_judgment.py --model-list Meta-Llama-3.1-8B --parallel 2
python gen_judgment.py --model-list Meta-Llama-3.1-8B --parallel 5

# show scores
python show_result.py --model-list Meta-Llama-3.1-8B