import sys
from PyQt5.QtCore import Qt, QTimer, QPoint, QRectF
from PyQt5.QtGui import (
    QFont,
    QPainter,
    QColor,
    QMouseEvent,
    QFontMetrics,
    QPainterPath,
    QRegion,
    QGuiApplication,
)
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QDesktopWidget, QWidget
from ui.window_effect import WindowEffect
import threading


class Chat_Window(QMainWindow):
    '''文字显示窗口'''

    _startPos = None
    _endPos = None
    _tracking = False

    def __init__(self):
        super().__init__()

        self.lyrics = ""
        self.desktop = QDesktopWidget()
        self.screen_width = self.desktop.screenGeometry().width()
        self.screen_height = self.desktop.screenGeometry().height()
        self.window_width = self.width()
        self.window_height = self.height()
        self.init_ui()

    def init_ui(self):
        print("初始化窗口")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )  # 无边框，置顶，隐藏任务栏图标

        # 窗口透明
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 窗口 亚克力|Aero 材质
        self.windowEffect = WindowEffect()
        # self.windowEffect.setAcrylicEffect(int(self.winId()), gradientColor="F2F2F230")  # F2F2F2F0亮色
        self.windowEffect.setAeroEffect(int(self.winId()))

        self.setGeometry((self.screen_width - 400), (self.screen_height - 100), 100, 50)

        # ?(没效果) 设置窗口圆角
        # self.setMask(self.create_rounded_rect_mask(12))

        # * 透明控件 用以拖动窗口
        self.drag_widget = QWidget(self)
        self.drag_widget.setGeometry(0, 0, self.width(), self.height())
        self.drag_widget.setObjectName("DragWidget")
        self.drag_widget.setStyleSheet(
            "#DragWidget { background-color: rgba(0, 0, 0, 0.01); border-radius: 16px;}"
        )

        self.lyric_label = QLabel(self)
        self.lyric_label.setAlignment(Qt.AlignCenter)
        self.lyric_label.setFont(QFont("微软雅黑", 13))
        self.lyric_label.setStyleSheet("color: rgb(255, 255, 255);")

        self.timer = QTimer(self)

    def get_window_bottom_color(self):
        '''获取窗口下方区域的颜色值或其他方式进行判断
        这里假设窗口下方区域的颜色是通过获取屏幕底部中央像素点的颜色值来确定的
        可以使用GUI库提供的功能来获取屏幕底部中央像素点的颜色值
        例如，使用PyQt5的QScreen类来获取屏幕底部中央像素点的颜色值'''

        screen = QGuiApplication.primaryScreen()
        pixmap = screen.grabWindow(0)
        window_rect = self.geometry()
        pixel = pixmap.toImage().pixel(
            (window_rect.x() + window_rect.width()) // 2,
            (window_rect.y() + window_rect.height()) - 1,
        )
        color = QColor(pixel)
        # 将颜色值转换为RGB颜色空间的值
        r, g, b, _ = color.getRgb()
        # 计算颜色的亮度值
        brightness = (r * 299 + g * 587 + b * 114) / 1000

        # 判断颜色值
        if brightness > 168:
            self.lyric_label.setStyleSheet("color: black;")
        elif color == QColor("black"):
            self.lyric_label.setStyleSheet("color: white;")
        else:
            self.lyric_label.setStyleSheet("color: white;")

    # * -------------------------------------------------------------------------
    # * 设置文本内容
    def set_lyric(self, lyric):
        print("更新文本" + lyric)

        self.get_window_bottom_color()
        # 根据窗口下方的颜色条件，更改lyric_label的字体颜色

        self.lyrics = lyric
        self.lyric_label.setText(self.lyrics)
        # 获取文本内容行数
        line_count = self.lyric_label.text().count('\n') + 1

        # 调整窗口高度
        new_height = line_count * 60  # 每行高度为20（可根据需要调整）

        # 根据文本长度计算新的窗口宽度
        new_width = QFontMetrics(self.lyric_label.font()).width(self.lyrics)  # 计算文本宽度

        self.resize(new_width + 25, new_height)
        # 设置文本控件大小和窗口大小一致
        self.lyric_label.setGeometry(0, 0, self.width(), self.height())
        self.drag_widget.setGeometry(0, 0, self.width(), self.height())
        self.window_width = self.width()
        self.window_height = self.height()

    # * ----------------------------------------------------
    # * 窗口的鼠标事件处理方法
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._tracking:
            self._endPos = event.pos() - self._startPos
            new_pos = self.pos() + self._endPos

            # 获取屏幕可见区域
            screen_rect = self.desktop.availableGeometry()

            # 确保新位置不超出屏幕范围
            new_x = max(
                screen_rect.x(),
                min(new_pos.x(), (self.screen_width - self.window_width)),
            )
            new_y = max(
                screen_rect.y(),
                min(new_pos.y(), (self.screen_height - self.window_height)),
            )

            # 计算新的窗口右边界
            new_right = new_x + self.window_width

            # 如果新的右边界超出了屏幕可见范围，调整窗口的位置
            if (new_right - self.window_width) > screen_rect.right():
                new_x = screen_rect.right() - self.window_width

            self.move(new_x, new_y)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._startPos = QPoint(event.x(), event.y())
            self._tracking = True

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._startPos = None
            self._tracking = False
            self._endPos = None

    # * ----------------------------------------------------
    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def closeEvent(self, event):
        from app import close_window

        close_window()
        event.accept()

    # * ----------------------------------------------------
    # ?(没效果) 设置窗口圆角遮罩
    def create_rounded_rect_mask(self, radius):
        rect = QRectF(0, 0, self.width(), self.height())
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        region = QRegion(path.toFillPolygon().toPolygon())
        return region
