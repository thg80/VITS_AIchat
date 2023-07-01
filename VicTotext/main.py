import logging
import threading
import time
import wave

import fastasr
import keyboard
import soundfile as sf
import speech_recognition as sr
from colorama import Back, Fore, Style
from pyaudio import PyAudio, paInt16

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s:%(message)s')
logger = logging.getLogger() 
fh = logging.FileHandler(filename='logger.log',encoding="utf-8",mode='a')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(message)s',datefmt='%m-%d %I:%M:%S'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)

param_path = r'FastASR/models'
audio_path = r'output.wav'
CHUNK = 1024
FORMAT = paInt16  # 16bit编码格式
CHANNELS = 1  # 单声道
RATE = 16000  # 16000采样频率


def convert_audio_to_text():
    # 创建麦克风的识别器对象
    r = sr.Recognizer()

    # 打开麦克风并录制音频
    with sr.Microphone() as source:
        print("我在听，你现在可以开始说话啦！~~")
        audio = r.listen(source)

    # 识别语音
    text = ""
    try:
        text = r.recognize_google(audio, language='zh-CN', show_all=False)
        print("你：" + text)
        with open("chatHistory.txt", "a", encoding="utf-8") as file:
            file.write("你：" + text)
    except Exception as e:
        print("没听到，无法识别：" + str(e))
    return text

def audio_record(out_file,stop_event):
    p = PyAudio()
    # 创建音频流
    stream = p.open(format=FORMAT,  # 音频流wav格式
                    channels=CHANNELS,  # 单声道
                    rate=RATE,  # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []  # 录制的音频流
    # 录制音频数据
    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    # 录制完成
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将录制的音频数据写入文件
    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    logger.info("录制完成！")
    audio_to_text()
def audio_to_text():
    data, samplerate = sf.read(audio_path, dtype='int16')
    audio_len = data.size / samplerate
    logger.info(Fore.YELLOW + "Audio time is {}s. len is {}.".format(audio_len, data.size))
    if audio_len > 0.1:
        start_time = time.time()
        p = fastasr.Model(param_path, 3)
        end_time = time.time()
        logger.info("Model 初始化 takes {:.2}s.".format(end_time - start_time) + Style.RESET_ALL)
        start_time = time.time()
        p.reset()
        result = p.forward(data)
        end_time = time.time()
        logger.info(Fore.GREEN+'Result: "{}".'.format(result) + Style.RESET_ALL)
        logger.info(Fore.YELLOW + "Model 推理 takes {:.2}s.".format(end_time - start_time) + Style.RESET_ALL)
        return(result)
    else:
        logger.warning(Back.RED +"语音时间不足0.1s" + Style.RESET_ALL)

def convert():
    logger.info(Fore.YELLOW +"Converting..."+ Style.RESET_ALL)
    is_recording = False
    audio_thread = None
    stop_event = threading.Event()
    while True:
        if keyboard.is_pressed("ctrl+t"):
            if not is_recording:
                is_recording = True
                audio_thread = threading.Thread(target=audio_record, args=(audio_path,stop_event))
                audio_thread.start()
                logger.info("开始录制...")
                time.sleep(0.2)
        elif is_recording:
            is_recording = False
            stop_event.set()
            break
    

if __name__ == "__main__":
    convert()