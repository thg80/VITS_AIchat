import argparse
import asyncio
import configparser
import json
import logging
import os
import subprocess
import threading
import time
import wave

import fastasr
import keyboard
import poe
import soundfile as sf
import speech_recognition as sr
import torch
from colorama import Back, Fore, Style
from pyaudio import PyAudio, paInt16
from scipy.io import wavfile
from torch import LongTensor, no_grad

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence


_init_vits_model = False
hps_ms = None
device = None
net_g_ms = None
pattern = r"。！？\n,;:]+|[.?!]+(?=[\s\n]|$))"

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s:%(message)s')
logger = logging.getLogger() 
fh = logging.FileHandler(filename='logger.log',encoding="utf-8",mode='a')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(message)s',datefmt='%m-%d %I:%M:%S'))
logger.addHandler(fh)

poe.headers = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58",
  "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
  "Accept-Encoding": "gzip, deflate, br",
  "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7",
  "Dnt": "1",
  "Sec-Ch-Ua": "\"Not.A/Brand\";v=\"8\", \"Chromium\";v=\"114\", \"Microsoft Edge\";v=\"114\"",
  "Sec-Ch-Ua-Mobile": "?0",
  "Sec-Ch-Ua-Platform": "\"Windows\"",
  "Upgrade-Insecure-Requests": "1"
}
#poe.logger.setLevel(logging.INFO)

with open('config.json','r') as f:
    config = json.load(f)


client = poe.Client(config['Poe']['token'], config['Poe']['proxy'])


async def handle_result(task):
    status, audios, time = task.result()
    print("--------------------------------")
    print("VITS : ", status, time)
    wavfile.write("output.wav", audios[0], audios[1])
    play_audio("output.wav")

async def send_chatgpt_request(send_msg):
    sentence = ""
    vits_sentence = "" #以读句
    print("[AI]: ",flush=True)


    for chunk in client.send_message(config['Poe']['bot'],send_msg,with_chat_break=False):
        print(chunk["text_new"],end="",flush=True)
        sentence = sentence + chunk["text_new"]

        if sentence.endswith(".") or sentence.endswith(",") or sentence.endswith("。") or sentence.endswith("?") or sentence.endswith("!") or sentence.endswith("？") or sentence.endswith("！") or sentence.endswith("，"):
            #断句
            if vits_sentence != "":
                vits_sentence = sentence.strip(vits_sentence)
            else:
                vits_sentence = sentence
            sentence = ""
            task = asyncio.create_task(vits(vits_sentence, 0, 124, config['Vic']['vitsNoiseScale'], config['Vic']['vitsNoiseScaleW'], config['Vic']['vitsLengthScale']))
            while not task.done():
                await asyncio.sleep(0.1)

            status, audios, time = task.result()  # 获取任务的返回值
            print("VITS-", status, time)
            wavfile.write("output.wav", audios[0], audios[1])
            play_audio("output.wav")



def play_audio(audio_file_name):
    command = f'mpv.exe -vo null {audio_file_name}'
    subprocess.run(command, shell=True)

def init_vits_model():
    global hps_ms, device, net_g_ms

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--colab", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    device = torch.device(args.device)
    hps_ms = utils.get_hparams_from_file(config['Vic']['model_path']+r'/config.json')
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval().to(device)
    speakers = hps_ms.speakers
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(config['Vic']['model_path'] + config['Vic']['checkpoint_model'], net_g_ms, None)
    _init_vits_model = True

def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text

async def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    global over

    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
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
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"




async def start():
    while True:
        print("请输入 >")
        input_str = await asyncio.get_event_loop().run_in_executor(None, input, '')
        if "关闭AI" in input_str:
            client.send_chat_break("capybara")
            return
        
        await send_chatgpt_request(input_str)

async def main():
    if not _init_vits_model:
        init_vits_model()
    await asyncio.gather(start(),)
if __name__ == "__main__":
    asyncio.run(main())
    
    

