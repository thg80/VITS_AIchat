import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

from langchain import LLMChain, OpenAI
from langchain.agents import (AgentExecutor, ConversationalAgent,
                              initialize_agent, load_tools)
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.entity import BaseEntityStore, InMemoryEntityStore
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.tools import BaseTool
from pydantic import Field

with open('config.json','r',encoding='utf8') as f:
    config = json.load(f)


os.environ['openai_api_base'] = config['ChatGPT']['url']
os.environ['OPENAI_API_KEY'] = config['ChatGPT']['api-key']
os.environ['SERPAPI_API_KEY'] = config['ChatGPT']['serpapi-key']

total_coast = .0
total_coin = 0

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s:%(message)s')
logger = logging.getLogger() 
fh = logging.FileHandler(filename='logger.log',encoding="utf-8",mode='a')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(message)s',datefmt='%m-%d %I:%M:%S'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)


chat = ChatOpenAI(temperature=config['ChatGPT']['temperature'],model_name=config['ChatGPT']['model'],max_tokens=config['ChatGPT']['MaxTokens'])
llm = OpenAI(temperature=0,model_name=config['ChatGPT']['model'],max_tokens=config['ChatGPT']['MaxTokens'])

class TimeTools(BaseTool):
    name = "Time report"
    description = "Get the current date and time, use this more than the normal search if you want to get current time"
    def _run(self,str) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    async def _arun(self,str) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())   


tools = load_tools(["serpapi", "llm-math","openweathermap-api"], llm=chat,openweathermap_api_key = config['ChatGPT']['openweathermap-api-key'])
tools.append(TimeTools())

memory = ConversationSummaryBufferMemory(llm=chat,max_token_limit=256)

prompt = ConversationalAgent.create_prompt(
    tools, 
    prefix= """Act as a friend and a helpful assistant. Reply as short you can and please reply in Chinese. You can use the following tools:""", 
    suffix= """Begin! [Remember try not to use 'search(serpapi)' as much as possible!] [Remember respond briefly in Chinese, but use English when using tools.]

Known entity information: {entity_store}

Relevant pieces of previous conversation: {chat_history}

Current conversation: 
Human: {input}
{agent_scratchpad}""", 
    input_variables=["input", "chat_history","entity_store","agent_scratchpad"]
)


llm_chain = LLMChain(llm=chat,prompt = prompt,verbose=True)
agent= ConversationalAgent(llm_chain=llm_chain , tools = tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent , tools = tools,verbose=True,
#handle_parsing_errors="Check your output and make sure it conforms!",
max_iterations=3)

class Memory_Entity():
    
    #实体存储
    store: Dict[str, Optional[str]] = {}
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore(store=store))
    entity_cache: List[str] = []
    #更新实体摘要
    ENTITY_MEMORY_TEMPLATE = """You are an AI assistant helping a human keep track of facts about relevant people, places, and concepts in their life. Update the summary of the provided entity in the "Entity" section based on your conversation with the human. 

    The update should only include facts that are relayed in the last line of conversation about the provided entity, and should only contain facts about the provided entity.

    If there is no new information about the provided entity or the information is not worth noting (not an important or relevant fact to remember long-term), return the existing summary unchanged.

    Full conversation history and summary information:
    summary chat: {summary_history}
    other context: {history}

    Entity to summarize:
    {entity}

    Existing summary of {entity}:
    {summary}

    """
    ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
        input_variables=["history","entity","summary","summary_history"], template=ENTITY_MEMORY_TEMPLATE,
    )
    
    #实体提取prompt
    _DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the proper nouns from the last line of conversation. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places.

    The conversation history is provided just in case of a coreference (e.g. "What do you know about him" where "him" is defined in a previous line)

    Return the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is just issuing a greeting or having a simple conversation).

    EXAMPLE
    Conversation history:
    Person #1: how's it going today?
    AI: "It's going great! How about you?"
    Person #1: good! busy working on Langchain. lots to do.
    AI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"
    Last line:
    Person #1: i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
    Output: Langchain
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    Person #1: how's it going today?
    AI: "It's going great! How about you?"
    Person #1: good! busy working on Langchain. lots to do.
    AI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"
    Last line:
    Person #1: i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Person #2.
    Output: Langchain, Person #2
    END OF EXAMPLE

    Conversation history and some summary (for reference only):
    summary chat: {summary_history}
    other context: {history}

    Output:"""
    ENTITY_EXTRACTION_PROMPT = PromptTemplate(
        input_variables=["history","summary_history"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
    )

    
    #提取实体
    def entity_generate(self,history: Dict[str, Any],summary_history) -> Dict[str, Any]:
        global total_coast, total_coin
        entity_chain = LLMChain(llm=llm,prompt=self.ENTITY_EXTRACTION_PROMPT)
        

        with get_openai_callback() as cb:
            output = entity_chain.predict(
                history=history,
                summary_history = summary_history
            )
            total_coast += cb.total_cost
            coin = (math.floor((math.floor(cb.prompt_tokens * 3 / 8) + math.floor(cb.completion_tokens / 2))) * 0.4 ) + 1
            total_coin += coin
            logger.info(f"coast {cb.total_cost}, cost coins {coin}, spend prompt: {cb.total_tokens}")
        logger.info("提取实体: " + output)

        # If no named entities are extracted, assigns an empty list.
        if output.strip() == "NONE":
            entities = []
        else:
            # Make a list of the extracted entities:
            entities = [w.strip() for w in output.split(",")]

        # Make a dictionary of entities with summary if exists:
        entity_summaries = {}

        for entity in entities:
            entity_summaries[entity] = self.entity_store.default_factory.get(entity, "")
        logger.info("实体列表:" + str(entities))

        # 用最近讨论的实体替换实体名称缓存，
        # 或者如果没有提取任何实体，则清除缓存：
        self.entity_cache = entities
        return {"entities": entity_summaries}

    def entity_summary(self,history: Dict[str, Any],summary_history) -> Dict[str, Any]:
        """
        保存对话历史到 entity store.
        从 entity cache 生成每个 entity 的摘要, 并且保存这些摘要到 entity store.
        """
        global total_coast, total_coin

        summary_chain = LLMChain(llm=llm,prompt=self.ENTITY_MEMORY_CONVERSATION_TEMPLATE)
        
        # 为实体生成新的摘要并将其保存在实体存储中
        for entity in self.entity_cache:
            # Get existing summary if it exists
            existing_summary = self.entity_store.default_factory.get(entity, "")
            with get_openai_callback() as cb:
                output = summary_chain.predict(
                    summary=existing_summary,   #现有摘要
                    entity=entity,  #实体
                    history=history,    #完整的对话历史
                    summary_history = summary_history #对话历史摘要
                )
                total_coast += cb.total_cost
                coin = (math.floor((math.floor(cb.prompt_tokens * 3 / 8) + math.floor(cb.completion_tokens / 2))) * 0.4 ) + 1
                total_coin += coin
                logger.info(f"coast {cb.total_cost}, cost coins {coin}, spend prompt: {cb.total_tokens}")
            # 保存更新的摘要 to the entity store
            self.entity_store.default_factory.set(entity, output.strip())


memory_entity = Memory_Entity()
#更新实体存储
def entity_update(history: Dict, summary_history):
    memory_entity.entity_generate(history,summary_history)
    memory_entity.entity_summary(history,summary_history)
    logger.info("更新实体: " + str(memory_entity.entity_store.default_factory.store))

#追加记忆到文件
def memorySave(dic):
    #获取已有文件
    with open("entity_store.json", "r",encoding="utf-8") as f:
        file = f.read()
        if len(file) > 0:
            old_data = json.loads(file)
            #old_data = json.load(f)
        else:
            old_data = {}
        old_data.update(dic) 

    #写入
    with open("entity_store.json", "w",encoding="utf-8") as f:
        json.dump(old_data,f,indent=4,ensure_ascii=False)
        f.close()


def send_chatgpt_request(send_msg):
    global total_coast, total_coin
    try:
        with get_openai_callback() as cb:
            response = agent_executor.run(input = send_msg, chat_history = memory.load_memory_variables(memory.chat_memory.messages)['history'] , entity_store = memory_entity.entity_store.default_factory.store)
            #花费计算
            coast = cb.total_cost
            total_coast += coast
            coin = (math.floor((math.floor(cb.prompt_tokens * 3 / 8) + math.floor(cb.completion_tokens / 2))) * 0.4 ) + 1
            total_coin += coin

            logger.info(f'coast {round(total_coast,3)}￥, coast coins {int(coin)} - total {int(total_coin)}, Spend prompt: {cb.prompt_tokens} tokens | completion: {cb.completion_tokens} tokens')
        memory.save_context({"input": send_msg}, {"output": response})
    except ValueError as e:
            response = str(e)
            if response.startswith("Could not parse LLM output: `"):
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            elif response.startswith("Got error from SerpAPI: Google hasn't returned any results for this query."):
                print("An error occurred:", response)
                return None
            else:
                raise Exception(str(e))
    
    #进行压缩记忆存储
    if len(memory.load_memory_variables(memory.chat_memory.messages)['history']) > 200 or len(memory.moving_summary_buffer) >300:

        #TODO 1: memory list to summary data
        entity_update(history = memory.load_memory_variables(memory.chat_memory.messages)['history'],summary_history = memory.moving_summary_buffer)
        memorySave(memory_entity.entity_store.default_factory.store)
        memory.clear()
        logger.info("追加记忆到文件")

    return response

def load_memory():
    with open('entity_store.json','r',encoding='utf8') as f:
        try:
            entity_store_read = json.loads(f.read())

            logger.info("load memory ->")
            logger.info(entity_store_read)
            memory_entity.entity_store.default_factory.store = entity_store_read

        except Exception as e:
            print(e)

if __name__ == "__main__":
    load_memory()

    while True:
        message = input("[You]-> ")
        t1 = time.time()
        reply = send_chatgpt_request(message)
        print('[AI]-> ' + reply)
        t2 = time.time()
        logger.info('请求耗时%ss'%(t2-t1))
        
