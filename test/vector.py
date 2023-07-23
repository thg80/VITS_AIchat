import json
import os
import time

import faiss
from langchain import LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import (VectorStoreRetrieverMemory)
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

with open('config.json','r',encoding='utf8') as f:
    config = json.load(f)


os.environ['openai_api_base'] = config['ChatGPT']['url']
os.environ['OPENAI_API_KEY'] = config['ChatGPT']['api-key']
os.environ['SERPAPI_API_KEY'] = config['ChatGPT']['serpapi-key']

embedding = OpenAIEmbeddings()
embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})



retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.retriever.vectorstore.load_local("load",embedding)



memory.save_context({"input": "你是我的私人助手，你的名字是“Sero”"}, {"output": "ok"})
print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])
# memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
# memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"}) 

chat = ChatOpenAI(temperature=config['ChatGPT']['temperature'],model_name=config['ChatGPT']['model'],max_tokens=config['ChatGPT']['MaxTokens'])


prompt_template = '''The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. 
Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)
Current conversation:
Human:{input}
'''
PROMPT = PromptTemplate(
    input_variables=["history","input"],template=prompt_template
)


#vectorstore.save_local("load")



llm_chain = LLMChain(llm=chat,verbose=True,prompt=PROMPT,memory = memory)

def load_memory():
    with open('entity_store.json','r',encoding='utf8') as f:
        try:
            entity_store_read = json.loads(f.read())
            for key,value in  entity_store_read.items():
                print("[load memory] -> " + key + ": " + value)
                memory.save_context({"input": key},{"output": value})
            memory.retriever.vectorstore.save_local("load")
        except Exception as e:
            raise print(e)

load_memory()
while True:
    #获取输入
    input_ = input("Enter your query: ")
    result = llm_chain.predict(input = input_)
    print(result)

