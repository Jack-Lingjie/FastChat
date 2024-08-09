#!bin/bash
python -m vllm.entrypoints.openai.api_server --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype bfloat16 --api-key token-abc123
python -m vllm.entrypoints.openai.api_server --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --api-key token-abc123
python -m vllm.entrypoints.openai.api_server --model /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --max-model-len 2048
vllm serve /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --api-key token-abc123
vllm serve /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --dtype auto --api-key token-abc123 --chat-template ./examples/template_chatml.jinja
python fastchat/llm_judge/gen_api_answer.py --model Meta-Llama-3.1-8B 


python fastchat/llm_judge/gen_model_answer.py \
--model-path /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B \
--model-id llama3.1-8b-base \
# --num-gpus-total 4
python gen_model_answer.py --model-path /home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B --model-id llama3.1-8b-base --num-gpus-total 4