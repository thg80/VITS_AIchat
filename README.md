# VITS_AIchat

- 使用 [FastASR](https://github.com/chenkui164/FastASR) 实现语音识别
- 使用 https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai 实现语音合成

## 使用说明

##### Langchain

- 更改 **BASEconfig.json** -> **config.json** 并填入 api-key
- 前往 [SerpApi: Google Search API](https://serpapi.com/) 获取 **serpapi-key**
- (可选) 前往 [Members (openweathermap.org)](https://home.openweathermap.org/api_keys) 获取 **openweathermap-api-key**

## config 文件说明

| 参数            | 参数说明                |
| --------------- | ----------------------- |
| input_mode      | 0 文字输入 ; 1 语音输入 |
| vitsNoiseScale  | 控制感情变化程度        |
| vitsNoiseScaleW | 控制音素发音长度        |
| vitsLengthScale | 控制整体语速            |

## TODO

- [ ] 添加其他 api，暂定 https://aigptx.top/ （用法同 chatgpt）[ post 方法无法流式输出 ]

## 日志/记录

| 日期     | 说明           |
| -------- | -------------- |
| 2023.7.8 | poe 被封了 orz |
| 2023.7.9 | 测试 langchain |
