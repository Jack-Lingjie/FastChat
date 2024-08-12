from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="a",

)

completion = client.chat.completions.create(
  model="/home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B-Instruct",
  # model="/home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface/Meta-Llama-3.1-8B",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)