import json
import time

import requests

#原始对话
with open('config.json','r',encoding='utf8') as f:
    config = json.load(f)

def send_chatgpt_request(send_msg):
    data = {
        "model": config["Chatgpt"]["model"],
        "messages": [
            {"role": 'system', 'content': config['Chatgpt']['InitPrompt']},
            {"role": "user", "content": send_msg}
        ],
        "max_tokens": config['Chatgpt']['MaxTokens'],
        "temperature": config['Chatgpt']['temperature']
    }
    data = json.dumps(data)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer "+ config['Chatgpt']['api-key']}

    try: 
        t1 = time.time()
        response = requests.post(config['Chatgpt']['url'], data=data, headers=headers,verify=False)
        response.raise_for_status()
    except Exception.Timeout as e:
        print('请求超时：'+str(e.message))
    except Exception.HTTPError as e:
        print("http请求错误:"+str(e.message))
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    else:
        print('请求错误：'+str(response.status_code)+','+str(response.reason))

    output = response.json()
    reply = output["choices"][0]["message"]['content']
    t2 = time.time()
    print('请求耗时%ss'%(t2-t1))
    print("[AI回复] : ", reply)
    return reply
    
if __name__ == "__main__":
    send_chatgpt_request("你可以做什么")