import ollama
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time

MODEL = 'llama3.1'
t0 = time.time()


def get_execution_time():
    return f"[{(time.time() - t0):.3f} s] "


print('Loading the documents')
# loader = DirectoryLoader("/home/felipe/Documents/Resume/", glob="**/*(en).pdf", show_progress=True)
# docs = loader.load()
# file_path = "https://www.ibm.com/investor/att/pdf/IBM_Annual_Report_2022.pdf"
file_path = "/home/felipe/Downloads/CS403-1.10-Database-Design-2nd-Edition-CCBY.pdf"
# file_path = "/home/felipe/Documents/Books/Design Patterns GoF.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

print(get_execution_time() +
      'Splitting - Needed for both information retrieval and downstream question-answering purposes')
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=100, add_start_index=True
# )
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(pages)

print(get_execution_time() + 'Creating Embeddings - Create a vector representation of a piece of text')
embeddings = OllamaEmbeddings(model=MODEL)

print(get_execution_time() + 'Creating Retriever for documents using Facebook AI Similarity Search (FAISS) ')
retriever = FAISS.from_documents(docs, embeddings).as_retriever()

print(get_execution_time() + 'Starting LLM')
llm = OllamaLLM(model=MODEL)

print(get_execution_time() + 'Creating Chain')
system_prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Question: {question}"
    "Context: {context}"
    "Answer:"
)
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
])

chain = ({
             "context": retriever,
             "question": RunnablePassthrough()
         }
         | prompt
         | llm
         | StrOutputParser()
         )

print(get_execution_time() + 'Extracting basic info')
answer = chain.invoke('Can you summarize this pdf for me? What is this pdf about? what is the main theme?')

print('\n' + get_execution_time() + 'Generated Text: \n')
print(answer)

while True:
    question = input("\n\n\nWhat else do you wish to know?\n")
    answer = chain.invoke(question)
    print('\n' + get_execution_time() + 'Answer: \n')
    print(answer)
