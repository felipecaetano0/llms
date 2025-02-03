from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import time

MODEL = "llama3.2"

question = input('Padawan Question: ')
t0 = time.time()

context = "You are master yoda, wise and helpful. Answer all questions to the best of your ability in portuguese"
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    # "Use three sentence maximum and keep the answer concise. "
    f"Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    # ('system', "You are a helpful assistant. Answer all questions to the best of your ability."),
    ('user', question)
])

model = OllamaLLM(model=MODEL)

chain = prompt | model

# response = chain.invoke({"question": "What is LangChain?"})
response = chain.invoke({"question": question})

t1 = time.time()

print(response)

print('Processing Time: ' + str(t1 - t0) + 's')
