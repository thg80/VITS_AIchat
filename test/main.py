import logging
import time

import lchain
import soundfile as sf
import speech_recognition as sr
from colorama import Back, Fore, Style

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s:%(message)s')
logger = logging.getLogger() 
fh = logging.FileHandler(filename='logger.log',encoding="utf-8",mode='a')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(message)s',datefmt='%m-%d %I:%M:%S'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)


def run():
    while True:
        message = input("[You]-> ")
        reply = lchain.send_chatgpt_request(message)
        print('[AI]-> ' + reply)

if __name__ == "__main__":
    lchain.load_memory()
    run()
    