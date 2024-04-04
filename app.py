# from ollama import Client
# client = Client(host='http://localhost:11434')
# response = client.chat(model='llama2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])

import ollama

stream = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Write a code for fibonacci series - write it 20 words'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)