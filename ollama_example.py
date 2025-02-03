from ollama import ChatResponse, chat
# import ollama

response: ChatResponse = chat(model='mistral', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

# ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])

print(response.message.content)
