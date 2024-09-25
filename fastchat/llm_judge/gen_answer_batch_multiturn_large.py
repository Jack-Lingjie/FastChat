from vllm import LLM, SamplingParams  
from tqdm import tqdm  
import datasets  
import json  
import argparse  
import shortuuid  
import time  
  
parser = argparse.ArgumentParser(description='Set model name.')  
parser.add_argument('--model-name', type=str, required=True, help='Name of the model to use')  
parser.add_argument('--model-path', type=str, default="/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/sft_merge_checkpoints", help='Dir of the model to use')  
args = parser.parse_args()  
path_dir = args.model_path  
model_name = args.model_name  
  
template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message + '\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ 'Human: ' + content + '\nAssistant:' }}{% elif message['role'] == 'assistant' %}{{ content + '<|end_of_text|>' + '\n' }}{% endif %}{% endfor %}"
 
llm = LLM(model=f"{path_dir}/{model_name}", tensor_parallel_size=4)
  
print(f"model name: {model_name}")  
print(f"model_path: {path_dir}/{model_name}")  
  
gen_kwargs_vllm = {
    "max_tokens": 2048,
    "temperature": 0.0,
}
tokenizer = llm.get_tokenizer()  
if tokenizer.chat_template is None:  
    tokenizer.chat_template = template  
    # print(f"tokenizer.eos_token_id:{tokenizer.eos_token_id}")
    # print(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    # print(tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
    # tokenizer.chat_template = tokenizer.chat_template.replace("<|eot_id|>", tokenizer.eos_token)  
    # print(tokenizer.chat_template)
    gen_kwargs_vllm['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    print(f"tokenizer.chat_template: {tokenizer.chat_template}")  
    print("tokenizer is None, use setted template")  
else:  
    gen_kwargs_vllm['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")]  
    print("use original template")  
  
sampling_params = SamplingParams(**gen_kwargs_vllm)  
  
# 加载question.jsonl数据集  
questions = datasets.load_dataset('json', data_files='data/mt_bench/question.jsonl')['train']  
  
def convert_to_message(example):  
    messages = [{"role": "user", "content": example["turns"][0]}]  
    example["messages"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  
    return example  
  
questions = questions.map(convert_to_message)  
  
# 生成第一轮输出  
encoded_inputs = tokenizer.batch_encode_plus(  
    questions['messages'],  
    add_special_tokens=False,
) 
input_ids = encoded_inputs['input_ids']  

outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
outputs_text = [x.outputs[0].text for x in outputs]  
token_lens = [len(x.outputs[0].token_ids) for x in outputs]  
  
# 第一轮对话结果  
questions = questions.remove_columns(["messages"])  
questions = questions.add_column("output_round_1", outputs_text)  
questions = questions.add_column("token_lens_round_1", token_lens)  
  
# 第二轮对话  
def second_round_messages(example):  
    messages = [  
        {"role": "user", "content": example["turns"][0]},  
        {"role": "assistant", "content": example["output_round_1"]},  
        {"role": "user", "content": example["turns"][1]}  
    ]  
    example["messages"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  
    return example  
  
questions = questions.map(second_round_messages)  
  
# 生成第二轮输出  
outputs = llm.generate(questions['messages'], sampling_params)  
encoded_inputs = tokenizer.batch_encode_plus(  
    questions['messages'],  
    add_special_tokens=False,
) 
input_ids = encoded_inputs['input_ids']  
outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
outputs_text = [x.outputs[0].text for x in outputs]  
token_lens = [len(x.outputs[0].token_ids) for x in outputs]  
  
# 移除现有的 'messages' 列并添加新的列  
questions = questions.remove_columns(["messages"])  
questions = questions.add_column("output_round_2", outputs_text)  
questions = questions.add_column("token_lens_round_2", token_lens)  
  
def transform_example(example):  
    turns = [  
        example['output_round_1'],  
        example['output_round_2']  
    ]  
    transformed_example = {  
        "question_id": example['question_id'],  
        "answer_id": shortuuid.uuid(),  
        "model_id": model_name,  
        "choices": [  
            {  
                "index": 0,  
                "turns": turns  
            }  
        ],  
        "tstamp": time.time()  
    }  
    return transformed_example  
  
# 对数据集进行转换  
transformed_dataset = questions.map(transform_example, remove_columns=questions.column_names)  
  
# 保存为 JSONL 格式  
with open(f'data/mt_bench/model_answer/{model_name}.jsonl', 'w', encoding='utf-8') as f:  
    for example in transformed_dataset:  
        json.dump(example, f)  
        f.write('\n')  

with open(f'/mnt/lingjiejiang/textual_aesthetics/data/mtbench/{model_name}.jsonl', 'w', encoding='utf-8') as f:  
    for example in transformed_dataset:  
        json.dump(example, f)  
        f.write('\n')    
print(f"Data saved to data/mt_bench/model_answer/{model_name}.jsonl")  
print(f"/mnt/lingjiejiang/textual_aesthetics/data/mtbench/{model_name}.jsonl")  
