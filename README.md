# VITS_AIchat

* 使用 [FastASR](https://github.com/chenkui164/FastASR) 实现语音识别
* 使用 https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai 实现语音合成


## config 文件说明

| 参数            | 参数说明              |
| --------------- | --------------------- |
| input_mode      | 0文字输入 ; 1语音输入 |
| vitsNoiseScale  | 控制感情变化程度      |
| vitsNoiseScaleW | 控制音素发音长度      |
| vitsLengthScale | 控制整体语速          |
