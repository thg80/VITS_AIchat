import argparse
import asyncio
import collections
import json
import logging
import os
import signal
import subprocess
import threading
import time
import wave
from array import array
from struct import pack

import keyboard
import soundfile as sf
import torch
import webrtcvad
from colorama import Back, Fore, Style
from pyaudio import PyAudio, paInt16
from torch import LongTensor, no_grad

import fastasr
import modules.commons
import modules.utils
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

vic2text_path = r"FastASR/models"
record_path = r"record.wav"

CHUNK = 1024
FORMAT = paInt16  # 16bit编码格式
CHANNELS = 1  # 单声道
RATE = 16000  # 16000采样频率


with open("config.json", "r") as f:
    config = json.load(f)


class MyThread(threading.Thread):
    def __init__(self, target=None, args=()):
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


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
    hps_ms = modules.utils.get_hparams_from_file(config["Vic"]["model_path"] + r"/config.json")
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model,
    )
    _ = net_g_ms.eval().to(device)
    speakers = hps_ms.speakers
    model, optimizer, learning_rate, epochs = modules.utils.load_checkpoint(
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




# 语音录制
def audio_record(out_file, stop_event):
    play_audio("trigger.wav")
    time.sleep(0.1)
    p = PyAudio()
    # 创建音频流
    stream = p.open(
        format=FORMAT,  # 音频流wav格式
        channels=CHANNELS,  # 单声道
        rate=RATE,  # 采样率16000
        input=True,
        frames_per_buffer=CHUNK,
    )
    frames = []  # 录制的音频流
    # 录制音频数据
    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)
        print("-", end="", flush=True)

    # 录制完成
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(">", end="", flush=True)
    # 将录制的音频数据写入文件
    wf = wave.open(out_file, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    logger.info("录制完成！")
    play_audio("done.wav")
    return audio_to_text()


# 转文字
def audio_to_text():
    data, samplerate = sf.read(record_path, dtype="int16")
    audio_len = data.size / samplerate
    logger.info(
        Fore.YELLOW + "Audio time is {}s. len is {}.".format(audio_len, data.size)
    )
    if audio_len > 0.1:
        start_time = time.time()
        p.reset()
        result = p.forward(data)
        end_time = time.time()
        logger.info(Fore.GREEN + 'Result: "{}".'.format(result) + Style.RESET_ALL)
        print(Fore.GREEN + 'Result: "{}".'.format(result) + Style.RESET_ALL)
        logger.info(
            Fore.YELLOW
            + "Model 推理 takes {:.2}s.".format(end_time - start_time)
            + Style.RESET_ALL
        )
        if result != "":
            return result
        else:
            logger.warning(Back.RED + "输入为空" + Style.RESET_ALL)
            return None
    else:
        logger.warning(Back.RED + "语音时间不足0.1s" + Style.RESET_ALL)
        return None


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
            print(Fore.YELLOW + "Converting..." + Style.RESET_ALL)
            is_recording = False
            audio_thread = None
            stop_event = threading.Event()
            while True:


                if keyboard.is_pressed(config["hotkey"]):
                    if not is_recording:
                        is_recording = True
                        audio_thread = MyThread(
                            target=audio_record, args=(record_path, stop_event)
                        )
                        audio_thread.start()
                        logger.info("开始录制...")
                        print(Fore.YELLOW + "开始录制..." + Style.RESET_ALL)

                elif is_recording:
                    is_recording = False
                    stop_event.set()
                    audio_thread.join()
                    result = audio_thread.get_result()
                    if result is not None:
                        if "切换输入" in result:
                            config["input_mode"] = 1
                            play_audio("done.wav")
                            break
                        await bot_runner(result, config["bot"])
                    break


async def main():
    if not _init_vits_model:
        init_vits_model()
    await asyncio.gather(
        start(),
    )

def init():
    global p
    #初始化语音识别mox
    start_time = time.time()
    p = fastasr.Model(vic2text_path, 3)
    end_time = time.time()
    logger.info(
        "语音识别模型初始化 takes {:.2}s.".format(end_time - start_time) + Style.RESET_ALL
    )

    from bots.bot import init_bot
    init_bot()

if __name__ == "__main__":
    init()
    asyncio.run(main())
