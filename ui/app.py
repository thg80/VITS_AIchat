# coding:utf-8
import sys

from PIL import Image
import pystray
from PyQt5.QtCore import Qt, QThreadPool, QRunnable, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

from ui.Ui_ChatWindow import Chat_Window
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

main = sys.modules["__main__"]
if_show = True


def open_setting(icon, item):
    print("open setting")


def pause_a_re():
    print("pause")


def show_or_off_reply_window():
    global window, task, if_show
    try:
        if not if_show:
            print("SHOW WINDOW")
            window.show()

            window.activateWindow()
            if_show = True

        else:
            print("HIDE WINDOW")
            window.hide()
            if_show = False
    except Exception as e:
        raise ("WINDOWS ERROR: " + str(e))


def text_send_thread(text):
    global window

    if not if_show:
        main.logger.info("window is hidden")
    else:
        if window.isHidden():
            window.show()
        window.set_lyric(text)
        print(len(text) * 0.5)

        timer = threading.Timer((len(text) * 0.5), window.hide)
        timer.start()


def hide_reply_window():
    global app, window
    window.hide()


def close_reply_window(app):
    global window
    window.close()
    app.quit()


# async def icon_thread():
#     # ?! 貌似会直接运行 且 window会堵塞进程
#     global window, app, loop

#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(window_thread(window, app))

#     print("icon thread")
#     image = Image.open("./resource/icon.png")
#     menu = (
#         pystray.MenuItem("设置", open_setting),
#         pystray.MenuItem("暂停/恢复 输入", pause_a_re),
#         pystray.MenuItem("显示回复悬浮窗", show_or_off_reply_window),
#         pystray.MenuItem("退出", lambda icon, item: icon.stop()),
#     )
#     icon = pystray.Icon("VitsAssistant", image, "VitsAssistant", menu)

#     await asyncio.to_thread(icon_run, icon)


async def win_icon_thread():
    global window
    print("win icon thread")

    image = Image.open("./resource/icon.png")
    menu = (
        pystray.MenuItem("设置", open_setting),
        pystray.MenuItem("暂停/恢复 输入", pause_a_re),
        pystray.MenuItem("显示回复悬浮窗", show_or_off_reply_window),
        pystray.MenuItem("退出", lambda icon, item: icon.stop()),
    )
    icon = pystray.Icon("VitsAssistant", image, "VitsAssistant", menu)

    await asyncio.gather(
        asyncio.to_thread(icon_run, icon),
    )


def window_run(window):
    window.show()
    window.activateWindow()


def icon_run(icon):
    print("icon run")
    icon.run()


def open_setting():
    print("open setting")


def pause_a_re():
    print("pause a reply")


# 创建系统图标
def ui_init():
    global app, window
    window = Chat_Window()
