import transformers  
import torch  
from tqdm import tqdm  
import datasets  
import json  
import argparse  
import shortuuid  
import time  
  
# Argument parsing  
parser = argparse.ArgumentParser(description='Set model name.')  
parser.add_argument('--model-name', type=str, required=True, help='Name of the model to use')  
parser.add_argument('--model-path', type=str, default="/home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface", help='Dir of the model to use')  
args = parser.parse_args()  
  
path_dir = args.model_path  
model_name = args.model_name  
  
# Load the tokenizer and model  
tokenizer = transformers.AutoTokenizer.from_pretrained(path_dir, trust_remote_code=True)  
model = transformers.AutoModelForCausalLM.from_pretrained(path_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)  
model.to("cuda")  
  
print(f"model name: {model_name}")  
print(f"model_path: {path_dir}/{model_name}")  
  
gen_kwargs_hf = {  
    "max_new_tokens": 2048,  
    "temperature": 0.0,  
    "do_sample": False,  
}  
  
gen_kwargs_hf['eos_token_id'] = tokenizer.convert_tokens_to_ids("<|end_of_text|>")  
gen_kwargs_hf['pad_token_id'] = tokenizer.eos_token_id  
  
def generate_response(prompt):  
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)  
    tokens = model.generate(**inputs, **gen_kwargs_hf)  
    prompt = prompt.replace("<|begin_of_text|>", "")  
    return tokenizer.decode(tokens[0], skip_special_tokens=True).replace(prompt, '')  
  
# 加载question.jsonl数据集  
questions = datasets.load_dataset('json', data_files='data/mt_bench/question.jsonl')['train']  
  
# template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message + '\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ 'Human: ' + content + '\nAssistant:' }}{% elif message['role'] == 'assistant' %}{{ content + '<|end_of_text|>' + '\n' }}{% endif %}{% endfor %}"  
  
# def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):  
#     from jinja2 import Template  
#     rendered = Template(template).render(messages=messages)  
#     if add_generation_prompt:  
#         rendered += '\nAssistant:'  
#     if tokenize:  
#         return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)  
#     return rendered  
  
def convert_to_message(example):  
    messages = [{"role": "user", "content": example["turns"][0]}]  
    example["messages"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  
    return example  
  
questions = questions.map(convert_to_message)  
  
# 生成第一轮输出  
outputs_text = []  
token_lens = []  
  
for example in tqdm(questions):  
    response = generate_response(example['messages'])  
    outputs_text.append(response)  
    token_lens.append(len(tokenizer(response)['input_ids']))  
  
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
outputs_text = []  
token_lens = []  
  
for example in tqdm(questions):  
    response = generate_response(example['messages'])  
    outputs_text.append(response)  
    token_lens.append(len(tokenizer(response)['input_ids']))  
  
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
  
print(f"Data saved to data/mt_bench/model_answer/{model_name}.jsonl")  