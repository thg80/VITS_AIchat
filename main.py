import argparse
import asyncio
import json
import logging
import subprocess
import threading
import time

import keyboard
import torch
from colorama import Back, Fore, Style
from pyaudio import PyAudio, paInt16
from torch import LongTensor, no_grad

import modules.commons
import modules.utils as utils
from modules.models import SynthesizerTrn
from text import text_to_sequence


_init_vits_model = False
hps_ms = None
device = None
net_g_ms = None
limitation = 500

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

record_path = r"record.wav"

CHUNK = 1024
FORMAT = paInt16  # 16bit编码格式
CHANNELS = 1  # 单声道
RATE = 16000  # 16000采样频率


with open("config.json", "r") as f:
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
    hps_ms = utils.get_hparams_from_file(
        config["Vic"]["model_path"] + r"/config.json"
    )
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
    if len(text) > 200 and limitation:
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



async def start():
    from bots.bot import bot_runner

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
            from modules.audio import listen,stream_stop
            play_audio("trigger.wav")
            print(Fore.YELLOW + "开始监听..." + Style.RESET_ALL)
            stop = False
            while True:
                if keyboard.is_pressed(config["hotkey"]) or stop:
                    logger.info(Fore.RED + "停止监听" + Style.RESET_ALL)
                    time.sleep(1)
                    while True:
                        if keyboard.is_pressed(config["hotkey"]):
                            logger.info(Fore.RED + "继续监听".format(config["hotkey"]) + Style.RESET_ALL)
                            stop = False
                            break
                        time.sleep(0.8)
                
                
                result = listen(record_path)
                stream_stop()
                play_audio("done.wav")

                if result is not None:
                    if "停止" in result:
                        stop = True
                        continue
                    if "切换输入" in result:
                        config["input_mode"] = 1
                        play_audio("done.wav")
                        break
                    await bot_runner(result, config["bot"])
                    time.sleep(0.8)  # 避免VITS被录入
                    play_audio("trigger.wav")


async def main():
    if not _init_vits_model:
        init_vits_model()
    await asyncio.gather(
        start(),
    )


def init():

    from bots.bot import init_bot
    from modules.audio import init as model_init

    model_init()
    init_bot()

    with open("speakers.json", "r",encoding="utf-8") as f:
        file = json.load(f)
        speaker = file["speakers"][config["Vic"]["speaker_id"]]
        logger.info(Fore.GREEN + "Speaker: {}".format(speaker) + Style.RESET_ALL)



if __name__ == "__main__":
    init()
    asyncio.run(main())
