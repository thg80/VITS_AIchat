import json
import math
import os
import time
import requests
from langchain import LLMChain
from langchain.agents import (AgentExecutor,ConversationalAgent,initialize_agent,load_tools,)
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory,ConversationSummaryBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

with open('config.json','r',encoding='utf8') as f:
    config = json.load(f)

os.environ['openai_api_base'] = config['Chatgpt']['url']
os.environ['OPENAI_API_KEY'] = config['Chatgpt']['api-key']
os.environ['SERPAPI_API_KEY'] = config['Chatgpt']['serpapi-key']

total_coast = .0
total_coin = 0
 
chat = ChatOpenAI(temperature=config['Chatgpt']['temperature'],model_name=config['Chatgpt']['model'],max_tokens=config['Chatgpt']['MaxTokens'])
tools = load_tools(["serpapi", "llm-math","openweathermap-api"], llm=chat,openweathermap_api_key = config['Chatgpt']['openweathermap-api-key'])

history = ChatMessageHistory()
#memory = ConversationBufferMemory()
memory = ConversationSummaryBufferMemory(llm=chat,max_token_limit=256)

prompt = ConversationalAgent.create_prompt(
    tools, 
    prefix= """Act as a friend and a helpful assistant. Reply as short you can and please reply in Chinese. You can use the following tools:""", 
    suffix= """Begin! [Remember try not to use 'search(serpapi)' as much as possible!] [Remember respond briefly in Chinese, but use English when using tools.]

Previous conversation history: {chat_history}

Current conversation: {input}
{agent_scratchpad}""", 
    input_variables=["input", "chat_history","agent_scratchpad"]
)


llm_chain = LLMChain(llm=chat,prompt = prompt,verbose=True)
agent= ConversationalAgent(llm_chain=llm_chain , tools = tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent , tools = tools,verbose=True,handle_parsing_errors="Check your output and make sure it conforms!",max_iterations=3)

dicts = messages_to_dict(history.messages)
memory.save_context({"input": "我叫小明，请在回复的时候加上我的名字"}, {"output": "OK"})

def send_chatgpt_request(send_msg):
    global total_coast, total_coin

    try:
        with get_openai_callback() as cb:
            response = agent_executor.run(input = send_msg, chat_history = memory.load_memory_variables(memory.chat_memory.messages))
            #花费计算
            coast = cb.total_cost
            total_coast += coast
            coin = (math.floor((math.floor(cb.prompt_tokens * 3 / 8) + math.floor(cb.completion_tokens / 2))) * 0.4 ) + 1
            total_coin += coin

            print(f'coast {round(total_coast,3)}￥, coast coins {int(coin)} - total {int(total_coin)}, Spend prompt: {cb.prompt_tokens} tokens | completion: {cb.completion_tokens} tokens')
    except ValueError as e:
            response = str(e)
            if response.startswith("Could not parse LLM output: `"):
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            elif response.startswith("Got error from SerpAPI: Google hasn't returned any results for this query."):
                print("An error occurred:", response)
                return None
            else:
                raise Exception(str(e))


    memory.save_context({"input": message}, {"output": response})
    
    return response

if __name__ == "__main__":
    while True:
        message = input("[You]-> ")
        t1 = time.time()
        reply = send_chatgpt_request(message)
        print('[AI]-> ' + reply)
        t2 = time.time()
        print('请求耗时%ss'%(t2-t1))
