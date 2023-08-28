import argparse
import asyncio
import json
import logging
import subprocess
import time
import os
from pynput import keyboard
import torch
from colorama import Back, Fore, Style
from torch import LongTensor, no_grad
import re
import modules.commons
import modules.utils as utils
from modules.models import SynthesizerTrn
from text import text_to_sequence
from scipy.io import wavfile
import webbrowser
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

_init_vits_model = False
hps_ms = None
device = None
net_g_ms = None
# 按键停止监控与恢复
paused = False


record_path = r"record.wav"


# logger 相关
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")
logger = logging.getLogger()
fh = logging.FileHandler(filename="logger.log", encoding="utf-8", mode="a")
fh.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s:%(message)s", datefmt="%m-%d %I:%M:%S"
    )
)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def play_audio(audio_file_name):
    command = f"mpv.exe -vo null {audio_file_name}"
    subprocess.run(command, shell=True)


def init_vits_model():
    global hps_ms, device, net_g_ms, _init_vits_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--api", action="store_true", default=False)
    parser.add_argument(
        "--share", action="store_true", default=False, help="share gradio app"
    )
    parser.add_argument(
        "--colab", action="store_true", default=False, help="share gradio app"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    hps_ms = utils.get_hparams_from_file(config["Vic"]["model_path"] + r"/config.json")
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model,
    )
    _ = net_g_ms.eval().to(device)
    speakers = hps_ms.speakers
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(
        config["Vic"]["model_path"] + config["Vic"]["checkpoint_model"], net_g_ms, None
    )
    _init_vits_model = True


def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = modules.commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


async def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    global over

    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace("\n", " ").replace("\r", "").replace(" ", "")
    if len(text) > 250:
        return f"输入文字过长！{len(text)}>100", None, None
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        speaker_id = LongTensor([speaker_id]).to(device)
        audio = (
            net_g_ms.infer(
                x_tst,
                x_tst_lengths,
                sid=speaker_id,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"


# *快捷打开软件
def quick_launch(program):
    # 在配置文件中
    for programs in config["QuickLaunch"]:
        if "网页" in program and program in programs:
            webbrowser.open(config["QuickLaunch"][program])
            return True

        if program in programs:
            os.startfile(config["QuickLaunch"][program])
            return True

    return False


# * 其他语音操作
def extract_text(text):
    match = re.search(r'(?:打开|启动)\s*([^\s]+)', text)
    if match:
        program = match.group(1)
        if quick_launch(program):
            text = "已打开" + program
        else:
            text = "未找到" + program
        logger.info(Fore.YELLOW + text + Style.RESET_ALL)
        return text

    # 网页相关
    else:
        match = re.search(r'(?:哔哩哔哩搜索)\s*([^\s]+)', text)
        if match:
            search = match.group(1)
            url = f"https://search.bilibili.com/all?keyword={search}"
            webbrowser.open(url)
            logger.info(Fore.BLUE + "打开网页:" + url + Style.RESET_ALL)
            return "已打开网页"
        else:
            match = re.search(r'(?:搜索|浏览器搜索)\s*([^\s]+)', text)
            if match:
                search = match.group(1)
                url = f"https://www.bing.com/search?q={search}"
                webbrowser.open(url)
                logger.info(Fore.BLUE + "打开网页:" + url + Style.RESET_ALL)
                return "已打开网页"

    return None


async def play_hint_audio(text):
    '播放提示文本'
    status, audios, time = await vits(
        text,
        0,
        config["Vic"]["speaker_id"],
        config["Vic"]["vitsNoiseScale"],
        config["Vic"]["vitsNoiseScaleW"],
        config["Vic"]["vitsLengthScale"],
    )
    if status != "生成成功!":
        logger.error("VITS生成失败:", status)
    else:
        wavfile.write("output.wav", audios[0], audios[1])
        play_audio("output.wav")


# * 暂停监听
def on_press():
    '暂停监听'
    global paused
    paused = not paused
    if paused:
        logger.info(Fore.RED + "将在输出完成后停止监听" + Style.RESET_ALL)
    else:
        logger.info(Fore.GREEN + "恢复监听" + Style.RESET_ALL)
        play_audio("./resource/trigger.wav")
    time.sleep(0.1)


async def start():
    from bots.bot import bot_runner
    from ui.app import text_send_thread

    global paused

    listener = keyboard.GlobalHotKeys({"<ctrl>+t": on_press})
    listener.start()

    while True:
        # 文字输入
        if config["input_mode"]:
            print(Fore.LIGHTYELLOW_EX + "[You]-> ", end="" + Fore.RESET)
            input_str = await asyncio.get_event_loop().run_in_executor(None, input, "")
            if "切换输入" in input_str or "切换语音输入" in input_str:
                config["input_mode"] = 0
                continue
            await bot_runner(input_str, config["bot"])

        # 语音输入
        else:
            from modules.audio import listen, stream_stop

            play_audio("./resource/trigger.wav")
            await asyncio.sleep(1)
            print(Fore.YELLOW + "开始监听..." + Style.RESET_ALL)
            paused = False
            while True:
                if paused:
                    await asyncio.sleep(0.5)
                    continue
                else:
                    result = listen(record_path)
                    stream_stop()
                    play_audio("./resource/done.wav")

                    if result is not None and not paused:
                        if ("停止" or "暂停") in result:
                            paused = True
                            continue
                        if "切换输入" in result:
                            config["input_mode"] = 1
                            play_audio("./resource/done.wav")
                            break

                        # if "原神启动" in result:
                        #     import sys
                        #     import win32gui

                        #     os.startfile(
                        #         "F:\Game\Genshin Impact\Genshin Impact Game\YuanShen.exe"
                        #     )
                        #     while win32gui.FindWindow(None, "原神") == 0:
                        #         asyncio.sleep(0.5)
                        #         print("wait")
                        #     win32gui.SetForegroundWindow(
                        #         win32gui.FindWindow(None, "原神")
                        #     )
                        #     # 停止代码
                        #     print("stop")
                        #     sys.exit()

                        handled_text = extract_text(result)
                        if handled_text is not None:
                            text_send_thread(handled_text)
                            await play_hint_audio(handled_text)
                            await asyncio.sleep(0.8)
                            play_audio("./resource/trigger.wav")
                            continue

                        await bot_runner(result, config["bot"])
                        await asyncio.sleep(0.8)  # 避免VITS被录入
                        play_audio("./resource/trigger.wav")
                    pass


async def main():
    if not _init_vits_model:
        init_vits_model()
    from ui.app import icon_thread

    await asyncio.gather(icon_thread(), start())


def init():
    from bots.bot import init_bot
    from modules.audio import init as model_init
    from ui.app import ui_init

    start_time = time.time()

    with open("speakers.json", "r", encoding="utf-8") as f:
        file = json.load(f)
        speaker = file["speakers"][config["Vic"]["speaker_id"]]
        logger.info(Fore.GREEN + "Speaker: {}".format(speaker) + Style.RESET_ALL)

    model_init()
    init_bot()

    ui_init()

    end_time = time.time()
    print("初始化 总计花费 {:.2}s.".format(end_time - start_time) + Style.RESET_ALL)


if __name__ == "__main__":
    init()
    asyncio.run(main())
