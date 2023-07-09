import json
import os
import time

import requests
from langchain import LLMChain
from langchain.agents import (AgentExecutor, AgentType, ZeroShotAgent,
                              initialize_agent, load_tools)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

with open('config.json','r',encoding='utf8') as f:
    config = json.load(f)

os.environ['openai_api_base'] = config['Chatgpt']['url']
os.environ['OPENAI_API_KEY'] = config['Chatgpt']['api-key']
os.environ['SERPAPI_API_KEY'] = config['Chatgpt']['serpapi-key']

prefix = """Act as a friend and a helpful assistant. Reply as short you can and please reply in Chinese. You can use the following tools:"""
suffix = """[Remember try not to use 'search(serpapi)' as much as possible!] [Remember reply in Chinese]
Question: {input}
{agent_scratchpad}"""

chat = ChatOpenAI(temperature=0.9,model_name="gpt-3.5-turbo")
tools = load_tools(["serpapi", "llm-math","requests_all","openweathermap-api"], llm=chat,openweathermap_api_key = config['Chatgpt']['openweathermap-api-key'])
memory = ConversationBufferMemory(memory_key="chat_history")

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=chat,prompt = prompt)
tool_names = [tool.name for tool in tools]
agent=ZeroShotAgent(llm_chain=llm_chain , allowed_tools=tool_names)
# An agent with the tools, the language model, and the type of agent we want to use.
# agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,memory = memory)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent , tools = tools,verbose=True)

def send_chatgpt_request(send_msg):
    try:
        response = agent_executor.run(send_msg)
    except Exception as e:
            response = str(e)
            if response.startswith("Could not parse LLM output: `"):
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
                return response
            else:
                raise Exception(str(e))
    return response
    

if __name__ == "__main__":
    t1 = time.time()
    print(send_chatgpt_request("你打算做什么"))
    t2 = time.time()
    print('请求耗时%ss'%(t2-t1))