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

## 常见问题

- `pip install pyopenjtalk` 出错 - [参考](https://blog.csdn.net/ky1in93/article/details/129698278)

  将 `python`编译器添加到**系统环境变量**中，注意是将 `Visual Studio`的 `CMake\bin`添加到系统环境变量中！
  如果仍然无法安装，请 `pip list`检查一下是否安装了 `cmake`包。
  先 `pip uninstall cmake`卸载 `cmake`， 再安装。

## TODO

- [x] 添加其他 api，暂定 [aigptx.top](https://aigptx.top?aff=IfyQEDPv)
- [x] 使用 Lainchain 进行记忆存储
- [x] 压缩记忆初步实现（存为 ConversationSummaryBufferMemory 再存为实体）
- [x] 加入源项目
- [x] (-) bot 切换
- [x] 实时语音识别
- [ ] 无意义文本过滤

## 日志/记录

| 日期      | 说明                                          |
| --------- | --------------------------------------------- |
| 2023.7.8  | poe 被封了 orz                                |
| 2023.7.9  | 测试 langchain                                |
| 2023.7.10 | 使用 ConversationSummaryBufferMemory          |
| 2023.7.13 | 终于能存实体了 www<br />自建 Memory_Entity 类 |
| 2023.7.14 | 加入 langchain 到主项目                       |
| 2023.7.21 | 测试 VectorStoreRetrieverMemory               |
| 2023.7.23 | 加入 VectorStoreRetrieverMemory 到主项目      |
| 2023.7.24 | 拆分 bot 文件                                 |
| 2023.8.24 | 实时语音识别                                  |
