import poe
from colorama import Back, Fore, Style
import asyncio
from scipy.io import wavfile
from main import config, logger, vits, play_audio


poe.headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7",
    "Dnt": "1",
    "Sec-Ch-Ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Microsoft Edge";v="114"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Upgrade-Insecure-Requests": "1",
}
client = poe.Client(config["Poe"]["token"], config["Poe"]["proxy"])


def clean_chat():
    client.send_chat_break("capybara")


async def send_poe_request(send_msg):
    sentence = ""
    vits_sentence = ""  # 以读句
    print("[AI]: ", end="", flush=True)

    for chunk in client.send_message(
        config["Poe"]["bot"], send_msg, with_chat_break=False
    ):
        print(Fore.GREEN + chunk["text_new"], end="", flush=True)
        sentence = sentence + chunk["text_new"]

        if (
            sentence.endswith(".")
            or sentence.endswith("。")
            or sentence.endswith("?")
            or sentence.endswith("!")
            or sentence.endswith("？")
            or sentence.endswith("！")
            or sentence.endswith(":")
            or sentence.endswith("：")
        ):
            # 断句
            if sentence == "":
                continue
            if vits_sentence != "":
                vits_sentence = sentence.strip(vits_sentence)
            else:
                vits_sentence = sentence
            sentence = ""
            task = asyncio.create_task(
                vits(
                    vits_sentence,
                    0,
                    config["Vic"]["speaker_id"],
                    config["Vic"]["vitsNoiseScale"],
                    config["Vic"]["vitsNoiseScaleW"],
                    config["Vic"]["vitsLengthScale"],
                )
            )
            while not task.done():
                await asyncio.sleep(0.05)

            status, audios, time = task.result()  # 获取任务的返回值
            if task.result() == None:
                logger.warning(task.result())
                continue
            # print("VITS-", status, time)
            if task.done():
                print(Fore.RESET)
                logger.info("VITS-{}{}".format(status, time))
                wavfile.write("output.wav", audios[0], audios[1])
                play_audio("output.wav")
    print()
    return
