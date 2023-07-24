import os
from colorama import Back, Fore, Style
import asyncio
from scipy.io import wavfile
import sys

main = sys.modules["__main__"]


# 初始化bot
def init_bot():
    from bots.chatgpt_ import load_memory

    load_memory()


async def bot_runner(input_str, bot_name):
    match bot_name:
        case "ChatGPT":
            from bots.chatgpt_ import send_chatgpt_request

            output = send_chatgpt_request(input_str)
            print(Fore.LIGHTYELLOW_EX + "[AI]-> " + output + Fore.RESET)

            if len(output) > 200:
                # 切割output变量里面的句子
                output_split = output.split(".|。|？|?")
                if output_split is None:
                    return
                main.logger.info(f"切割回复 - {str(output_split)}")
                for sentence in output_split:
                    if output_split is None:
                        continue
                    status, audios, time = await main.vits(
                        sentence,
                        0,
                        main.config["Vic"]["speaker_id"],
                        main.config["Vic"]["vitsNoiseScale"],
                        main.config["Vic"]["vitsNoiseScaleW"],
                        main.config["Vic"]["vitsLengthScale"],
                    )
            else:
                status, audios, time = await main.vits(
                    output,
                    0,
                    main.config["Vic"]["speaker_id"],
                    main.config["Vic"]["vitsNoiseScale"],
                    main.config["Vic"]["vitsNoiseScaleW"],
                    main.config["Vic"]["vitsLengthScale"],
                )

            if status != "生成成功!":
                main.logger.error("VITS生成失败:", status)
            else:
                wavfile.write("output.wav", audios[0], audios[1])
                main.play_audio("output.wav")
                main.logger.info(Fore.GREEN + f"VITS: {status} {time}")
                Fore.RESET

        case "poe":
            from bots.poe_ import send_poe_request, clean_chat

            if "清除对话" in input_str:
                clean_chat()
                main.play_audio("done.wav")
                return
            send_poe_request(input_str)
        case _:
            raise ValueError("Unknown bot: {}".format(input_str))
