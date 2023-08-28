# coding:utf-8
import sys

from PIL import Image
import pystray
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

from ui.Ui_ChatWindow import Chat_Window
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

main = sys.modules["__main__"]
global app, window
lock = asyncio.Lock()


def open_setting(icon, item):
    print("open setting")


def pause_a_re():
    print("pause")


async def show_reply_window(window, app):
    print("show reply window")
    window.show()
    window.activateWindow()

    app.exec_()


def show_or_off_reply_window():
    global window, app, loop, task
    try:
        if window.isHidden():
            print("SHOW WINDOW")
            if not loop.is_running():
                loop.run_until_complete(window_thread(window, app))
            else:
                print("线程以运行")
                window.show()

        else:
            print("HIDE WINDOW")
            window.hide()
    except Exception as e:
        raise ("WINDOWS ERROR: " + str(e))


def text_send_thread(text):
    global app, window
    if window.isHidden() or app is None:
        main.logger.info("window is hidden")
    else:
        window.set_lyric(text)


def hide_reply_window():
    global app, window
    if app is not None and window.isHidden():
        window.hide()


def close_reply_window():
    global window, app, loop
    window.close()
    loop.close()
    app.quit()


async def window_thread(window, app):
    await loop.run_in_executor(None, show_reply_window, window, app)


async def icon_thread():
    # ?! 貌似会直接运行 且 window会堵塞进程
    global window, app, loop

    loop = asyncio.get_event_loop()
    loop.run_until_complete(window_thread(window, app))

    print("icon thread")
    image = Image.open("./resource/icon.png")
    menu = (
        pystray.MenuItem("设置", open_setting),
        pystray.MenuItem("暂停/恢复 输入", pause_a_re),
        pystray.MenuItem("显示回复悬浮窗", show_or_off_reply_window),
        pystray.MenuItem("退出", lambda icon, item: icon.stop()),
    )
    icon = pystray.Icon("VitsAssistant", image, "VitsAssistant", menu)

    await asyncio.to_thread(icon_run, icon)


def icon_run(icon):
    print("icon run")
    icon.run()


# 创建系统图标
def ui_init():
    global app, window

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = Chat_Window()
