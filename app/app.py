import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QHBoxLayout,
    QFrame, QSizePolicy
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QFont, QIcon, QLinearGradient, QBrush,
    QRadialGradient, QPainterPath, QPolygonF
)
from PyQt5.QtCore import Qt, QTimer, QUrl, QRectF, QPointF
from PyQt5.QtMultimedia import QSoundEffect

from keras.models import load_model
import torch
import joblib
import numpy as np
import pandas as pd

import sys
import os
import asyncio
import websockets
import threading
import json

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))

def app_path(*parts):
    return os.path.join(APP_DIR, *parts)

def root_path(*parts):
    return os.path.join(ROOT_DIR, *parts)

sys.path.append(ROOT_DIR)
from env.shared_memory_physics import read_telemetry
from env.shared_memory_graphics import read_graphics
from Training_Data.r1 import ResNet1DTabular
from Training_Data.r1s import ResNet1D

model = load_model(root_path("Training_Data", "models", "0_simple_cnn_model.keras"))
scaler = joblib.load(root_path("Training_Data", "models", "0_simple_scaler.pkl"))

sys.path.append(root_path("Training_Data"))

connected_clients = set()
websocket_loop = None

async def websocket_handler(websocket):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        connected_clients.discard(websocket)

async def _broadcast_data(label_pred, slips):
    if connected_clients:
        data = {
            "label_pred": label_pred,
            "slips": slips
        }
        message = json.dumps(data)
        clients = list(connected_clients)
        results = await asyncio.gather(
            *(client.send(message) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                connected_clients.discard(client)

async def send_data(label_pred, slips):
    if websocket_loop is None or not websocket_loop.is_running():
        await _broadcast_data(label_pred, slips)
        return

    current_loop = asyncio.get_running_loop()
    if current_loop is websocket_loop:
        await _broadcast_data(label_pred, slips)
        return

    future = asyncio.run_coroutine_threadsafe(
        _broadcast_data(label_pred, slips),
        websocket_loop,
    )
    await asyncio.wrap_future(future)

def start_websocket_server():
    global websocket_loop
    loop = asyncio.new_event_loop()
    websocket_loop = loop
    asyncio.set_event_loop(loop)

    async def run_server():
        server = await websockets.serve(websocket_handler, "localhost", 8765)
        print("WebSocket server avviato su ws://localhost:8765")
        await server.wait_closed()

    # Esegui la coroutine
    loop.run_until_complete(run_server())
    loop.run_forever()

LABEL_STATES = {
    0: ("Grip Loss", QColor(255, 58, 69)),
    1: ("High Grip - Accelerate", QColor(45, 240, 124)),
    2: ("Low Grip", QColor(255, 176, 48)),
    3: ("Neutral", QColor(93, 214, 255)),
}

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 1920
WINDOW_SCREEN_FILL = 0.94
MIN_WINDOW_WIDTH = 520
MIN_WINDOW_HEIGHT = 680

NEUTRAL_LABEL = 3
MIN_CLASSIFICATION_SPEED_KMH = 20.0
STEER_NEUTRAL_THRESHOLD = float(os.getenv("STEER_NEUTRAL_THRESHOLD", "0.03"))


def should_force_neutral(telem):
    return (
        telem["speed"] < MIN_CLASSIFICATION_SPEED_KMH
        or abs(telem["steer"]) < STEER_NEUTRAL_THRESHOLD
    )


def interpolate_color(value, max_val=1.0):
    ratio = min(max(value / max_val, 0.0), 1.0)
    hue = (120 - 120 * ratio) / 360.0 
    color = QColor.fromHsvF(hue, 0.85, 0.9) 
    return color

class WheelVisualizer(QWidget):
    wheel_labels = ("FL", "FR", "RL", "RR")

    def __init__(self):
        super().__init__()
        self.slips = [0.0] * 4
        self.setObjectName("GripVisualizer")
        self.setMinimumSize(280, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_slips(self, slips):
        self.slips = slips
        self.update()

    def _alpha(self, color, alpha):
        softened = QColor(color)
        softened.setAlpha(alpha)
        return softened

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        rect = QRectF(0, 0, w, h)

        background = QLinearGradient(0, 0, 0, h)
        background.setColorAt(0.0, QColor("#101215"))
        background.setColorAt(0.55, QColor("#171a1f"))
        background.setColorAt(1.0, QColor("#0a0c0f"))
        painter.fillRect(rect, QBrush(background))

        vignette = QRadialGradient(QPointF(w * 0.5, h * 0.42), max(w, h) * 0.72)
        vignette.setColorAt(0.0, QColor(255, 255, 255, 18))
        vignette.setColorAt(0.58, QColor(255, 255, 255, 4))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 132))
        painter.fillRect(rect, QBrush(vignette))

        grid_step = max(64, int(min(w, h) * 0.075))
        painter.setPen(QPen(QColor(255, 255, 255, 13), 1))
        for x in range(grid_step, w, grid_step):
            painter.drawLine(x, 0, x, h)
        for y in range(grid_step, h, grid_step):
            painter.drawLine(0, y, w, y)

        painter.setPen(QPen(QColor(93, 214, 255, 48), 2, Qt.DashLine))
        painter.drawLine(QPointF(w * 0.5, h * 0.08), QPointF(w * 0.5, h * 0.92))

        panel_margin = max(22, int(min(w, h) * 0.025))
        painter.setPen(QPen(QColor(255, 255, 255, 38), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(panel_margin, panel_margin, -panel_margin, -panel_margin), 30, 30)

        center_x = w * 0.5
        center_y = h * 0.53
        car_w = max(110, min(w * 0.26, h * 0.23, 320))
        car_h = max(260, min(h * 0.68, 760))
        car_top = center_y - car_h / 2
        car_bottom = center_y + car_h / 2
        wheel_r = max(28, min(w * 0.085, h * 0.075, 96))
        front_y = car_top + car_h * 0.24
        rear_y = car_top + car_h * 0.76
        left_x = center_x - car_w * 0.94
        right_x = center_x + car_w * 0.94

        axle_pen = QPen(QColor(139, 155, 169, 130), max(7, int(wheel_r * 0.08)), Qt.SolidLine, Qt.RoundCap)
        painter.setPen(axle_pen)
        painter.drawLine(QPointF(left_x, front_y), QPointF(right_x, front_y))
        painter.drawLine(QPointF(left_x, rear_y), QPointF(right_x, rear_y))

        wing_h = max(34, car_h * 0.035)
        front_wing = QRectF(center_x - car_w * 1.06, car_top - wing_h * 0.5, car_w * 2.12, wing_h)
        rear_wing = QRectF(center_x - car_w * 1.1, car_bottom - wing_h * 0.5, car_w * 2.2, wing_h)
        wing_gradient = QLinearGradient(front_wing.left(), 0, front_wing.right(), 0)
        wing_gradient.setColorAt(0.0, QColor("#2f3942"))
        wing_gradient.setColorAt(0.5, QColor("#59656f"))
        wing_gradient.setColorAt(1.0, QColor("#2f3942"))
        painter.setPen(QPen(QColor(150, 166, 178, 130), 1))
        painter.setBrush(QBrush(wing_gradient))
        painter.drawRoundedRect(front_wing, 10, 10)
        painter.drawRoundedRect(rear_wing, 10, 10)

        body_rect = QRectF(center_x - car_w * 0.33, car_top + car_h * 0.12, car_w * 0.66, car_h * 0.76)
        body_gradient = QLinearGradient(body_rect.left(), body_rect.top(), body_rect.right(), body_rect.bottom())
        body_gradient.setColorAt(0.0, QColor("#30363d"))
        body_gradient.setColorAt(0.45, QColor("#171b20"))
        body_gradient.setColorAt(1.0, QColor("#0c0f13"))
        body_path = QPainterPath()
        body_path.addRoundedRect(body_rect, 42, 42)
        painter.setPen(QPen(QColor(93, 214, 255, 118), 2))
        painter.setBrush(QBrush(body_gradient))
        painter.drawPath(body_path)

        nose = QPolygonF([
            QPointF(center_x, car_top + car_h * 0.035),
            QPointF(center_x - car_w * 0.18, car_top + car_h * 0.22),
            QPointF(center_x - car_w * 0.10, car_top + car_h * 0.53),
            QPointF(center_x + car_w * 0.10, car_top + car_h * 0.53),
            QPointF(center_x + car_w * 0.18, car_top + car_h * 0.22),
        ])
        painter.setPen(QPen(QColor(220, 230, 238, 70), 1))
        painter.setBrush(QColor("#222830"))
        painter.drawPolygon(nose)

        cockpit = QRectF(center_x - car_w * 0.18, center_y - car_h * 0.08, car_w * 0.36, car_h * 0.16)
        cockpit_gradient = QRadialGradient(cockpit.center(), cockpit.width() * 0.65)
        cockpit_gradient.setColorAt(0.0, QColor("#5dd6ff"))
        cockpit_gradient.setColorAt(0.42, QColor("#1f4452"))
        cockpit_gradient.setColorAt(1.0, QColor("#0b0d10"))
        painter.setPen(QPen(QColor(255, 255, 255, 88), 1))
        painter.setBrush(QBrush(cockpit_gradient))
        painter.drawEllipse(cockpit)

        accent_pen = QPen(QColor(255, 58, 69, 180), max(4, int(w * 0.005)), Qt.SolidLine, Qt.RoundCap)
        painter.setPen(accent_pen)
        painter.drawLine(QPointF(center_x, car_top + car_h * 0.16), QPointF(center_x, car_bottom - car_h * 0.15))

        wheel_centers = [
            QPointF(left_x, front_y),
            QPointF(right_x, front_y),
            QPointF(left_x, rear_y),
            QPointF(right_x, rear_y),
        ]

        for i, center in enumerate(wheel_centers):
            slip = self.slips[i]
            color = interpolate_color(slip)
            ratio = min(max(slip, 0.0), 1.0)
            wheel_rect = QRectF(center.x() - wheel_r, center.y() - wheel_r, wheel_r * 2, wheel_r * 2)

            glow = QRadialGradient(center, wheel_r * 1.9)
            glow.setColorAt(0.0, self._alpha(color, 120))
            glow.setColorAt(0.48, self._alpha(color, 38))
            glow.setColorAt(1.0, self._alpha(color, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(wheel_rect.adjusted(-wheel_r * 0.72, -wheel_r * 0.72, wheel_r * 0.72, wheel_r * 0.72))

            tire_gradient = QRadialGradient(center, wheel_r)
            tire_gradient.setColorAt(0.0, color.lighter(130))
            tire_gradient.setColorAt(0.55, color)
            tire_gradient.setColorAt(1.0, color.darker(165))
            painter.setBrush(QBrush(tire_gradient))
            painter.setPen(QPen(QColor(8, 10, 12), max(5, int(wheel_r * 0.08))))
            painter.drawEllipse(wheel_rect)

            painter.setPen(QPen(QColor(255, 255, 255, 65), max(2, int(wheel_r * 0.025))))
            painter.drawEllipse(wheel_rect.adjusted(wheel_r * 0.16, wheel_r * 0.16, -wheel_r * 0.16, -wheel_r * 0.16))

            arc_pen = QPen(color.lighter(135), max(8, int(wheel_r * 0.1)), Qt.SolidLine, Qt.RoundCap)
            painter.setPen(arc_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(wheel_rect.adjusted(-wheel_r * 0.18, -wheel_r * 0.18, wheel_r * 0.18, wheel_r * 0.18), 90 * 16, int(-360 * 16 * ratio))

            label_font = QFont("Segoe UI", max(15, int(wheel_r * 0.22)), QFont.Bold)
            painter.setFont(label_font)
            painter.setPen(QColor("#0a0d0f") if color.lightness() > 126 else QColor("#f8fafc"))
            painter.drawText(wheel_rect, Qt.AlignCenter, f"{slip:.2f}")

            tag_rect = QRectF(center.x() - wheel_r * 0.44, center.y() - wheel_r * 1.48, wheel_r * 0.88, wheel_r * 0.34)
            painter.setPen(QPen(QColor(255, 255, 255, 42), 1))
            painter.setBrush(QColor(11, 14, 18, 210))
            painter.drawRoundedRect(tag_rect, 8, 8)
            painter.setFont(QFont("Segoe UI", max(9, int(wheel_r * 0.12)), QFont.Bold))
            painter.setPen(QColor(226, 232, 240))
            painter.drawText(tag_rect, Qt.AlignCenter, self.wheel_labels[i])

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slip Dashboard")
        self._ui_scale = 1.0
        self._avg_color = interpolate_color(0.0)
        self._status_color = LABEL_STATES[NEUTRAL_LABEL][1]
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self.setWindowIcon(QIcon(app_path("img.png")))

        self.current_label_pred = None
        self.time_idx_counter = 0
        self.slip_window = [[], [], [], []]

        

        # Carica i suoni
        self.sounds = {
            0: QSoundEffect(),
            1: QSoundEffect(),
            2: QSoundEffect(),
            3: QSoundEffect()
        }
        self.sounds[1].setSource(QUrl.fromLocalFile(app_path("high_grip.wav")))
        self.sounds[2].setSource(QUrl.fromLocalFile(app_path("limit_grip.wav")))
        self.sounds[0].setSource(QUrl.fromLocalFile(app_path("loss_grip.wav")))

        for s in self.sounds.values():
            s.setVolume(0.5)

        self.setStyleSheet(self._base_stylesheet(self._ui_scale))

        self.header_panel = QFrame()
        self.header_panel.setObjectName("HeaderPanel")
        self.header_layout = QVBoxLayout(self.header_panel)
        self.header_layout.setContentsMargins(34, 30, 34, 30)
        self.header_layout.setSpacing(6)

        self.title_label = QLabel("SLIP DASHBOARD")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setFont(QFont("Segoe UI", 42, QFont.Black))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(False)

        self.subtitle_label = QLabel("ASSETTO CORSA REAL-TIME TELEMETRY")
        self.subtitle_label.setObjectName("SubtitleLabel")
        self.subtitle_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(False)

        self.header_layout.addWidget(self.title_label)
        self.header_layout.addWidget(self.subtitle_label)

        self.control_panel = QFrame()
        self.control_panel.setObjectName("ControlPanel")
        self.control_layout = QHBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(30, 24, 30, 24)
        self.control_layout.setSpacing(22)

        self.selector_label = QLabel("MODEL")
        self.selector_label.setObjectName("SelectorLabel")
        self.selector_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.selector_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Simple Model", "LSTM", "Transformers", "ResNet1D", "ResNet1D Sequence"])
        self.model_selector.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.model_selector.currentIndexChanged.connect(self.load_model)
        self.model_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.avg_label = QLabel("AVG SLIP 0.00")
        self.avg_label.setObjectName("AverageLabel")
        self.avg_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.avg_label.setAlignment(Qt.AlignCenter)
        self.avg_label.setMinimumWidth(210)
        self.avg_label.setWordWrap(False)

        self.control_layout.addWidget(self.selector_label)
        self.control_layout.addWidget(self.model_selector, stretch=1)
        self.control_layout.addWidget(self.avg_label)

        self.visualizer = WheelVisualizer()

        self.status_label = QLabel("READY")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setFont(QFont("Segoe UI", 34, QFont.Black))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(112)
        self.status_label.setWordWrap(False)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(42, 42, 42, 42)
        self.main_layout.setSpacing(24)
        self.main_layout.addWidget(self.header_panel)
        self.main_layout.addWidget(self.control_panel)
        self.main_layout.addWidget(self.visualizer, stretch=1)
        self.main_layout.addWidget(self.status_label)
        self.setLayout(self.main_layout)
        self.fit_to_available_screen()
        self.apply_responsive_sizing()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(150)

        self.model = None
        self.scaler = None
        self.window_size = 5
        self.realtime_window = []

        self.load_model()

    def _scaled(self, value, minimum=1):
        return max(minimum, int(round(value * self._ui_scale)))

    def _base_stylesheet(self, scale):
        def px(value, minimum=1):
            return max(minimum, int(round(value * scale)))

        arrow_path = app_path("arrow.svg").replace("\\", "/")
        return f"""
            QWidget {{
                background-color: #07090c;
                color: #f8fafc;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}

            QFrame#HeaderPanel,
            QFrame#ControlPanel {{
                background-color: #12161b;
                border: 1px solid #2d3640;
                border-radius: {px(26, 12)}px;
            }}

            QLabel#TitleLabel {{
                background-color: transparent;
                color: #f8fafc;
            }}

            QLabel#SubtitleLabel,
            QLabel#SelectorLabel {{
                background-color: transparent;
                color: #94a3b8;
            }}

            QLabel#AverageLabel {{
                background-color: rgba(93, 214, 255, 24);
                border: 1px solid rgba(93, 214, 255, 120);
                border-radius: {px(18, 9)}px;
                color: #dff7ff;
                padding: {px(16, 8)}px {px(22, 10)}px;
            }}

            QComboBox {{
                border: 1px solid #3b4652;
                border-radius: {px(16, 8)}px;
                padding: {px(16, 7)}px {px(52, 30)}px {px(16, 7)}px {px(20, 10)}px;
                background-color: #0d1117;
                font-size: {px(22, 12)}px;
                color: #f8fafc;
                min-height: {px(54, 32)}px;
            }}

            QComboBox:hover {{
                border-color: #5dd6ff;
                background-color: #111821;
            }}

            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: {px(48, 28)}px;
                border-left: 1px solid #2d3640;
                border-top-right-radius: {px(16, 8)}px;
                border-bottom-right-radius: {px(16, 8)}px;
            }}

            QComboBox::down-arrow {{
                image: url({arrow_path});
                width: {px(18, 10)}px;
                height: {px(18, 10)}px;
                margin-right: {px(15, 7)}px;
            }}

            QComboBox QAbstractItemView {{
                border: 1px solid #3b4652;
                selection-background-color: #1b6f8c;
                selection-color: white;
                padding: {px(8, 4)}px;
                background-color: #0d1117;
                color: #f8fafc;
                outline: none;
            }}
        """

    def fit_to_available_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
            return

        available = screen.availableGeometry()
        max_w = max(MIN_WINDOW_WIDTH, int(available.width() * WINDOW_SCREEN_FILL))
        max_h = max(MIN_WINDOW_HEIGHT, int(available.height() * WINDOW_SCREEN_FILL))

        frame_extra_w = max(16, self.frameGeometry().width() - self.geometry().width())
        frame_extra_h = max(48, self.frameGeometry().height() - self.geometry().height())
        target_w = max(MIN_WINDOW_WIDTH, min(WINDOW_WIDTH, max_w - frame_extra_w))
        target_h = max(MIN_WINDOW_HEIGHT, min(WINDOW_HEIGHT, max_h - frame_extra_h))
        self.resize(target_w, target_h)

        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        if frame.left() < available.left():
            frame.moveLeft(available.left())
        if frame.top() < available.top():
            frame.moveTop(available.top())
        if frame.right() > available.right():
            frame.moveRight(available.right())
        if frame.bottom() > available.bottom():
            frame.moveBottom(available.bottom())
        self.move(frame.topLeft())

    def apply_responsive_sizing(self):
        if not hasattr(self, "main_layout"):
            return

        self._ui_scale = max(
            0.52,
            min(1.0, min(self.width() / WINDOW_WIDTH, self.height() / WINDOW_HEIGHT)),
        )
        self.setStyleSheet(self._base_stylesheet(self._ui_scale))

        outer = self._scaled(42, 18)
        self.main_layout.setContentsMargins(outer, outer, outer, outer)
        self.main_layout.setSpacing(self._scaled(24, 10))

        self.header_layout.setContentsMargins(
            self._scaled(34, 16),
            self._scaled(30, 12),
            self._scaled(34, 16),
            self._scaled(30, 12),
        )
        self.header_layout.setSpacing(self._scaled(6, 2))

        self.control_layout.setContentsMargins(
            self._scaled(30, 14),
            self._scaled(24, 10),
            self._scaled(30, 14),
            self._scaled(24, 10),
        )
        self.control_layout.setSpacing(self._scaled(22, 8))

        self.title_label.setFont(QFont("Segoe UI", self._scaled(42, 22), QFont.Black))
        self.subtitle_label.setFont(QFont("Segoe UI", self._scaled(16, 9), QFont.Bold))
        self.selector_label.setFont(QFont("Segoe UI", self._scaled(15, 9), QFont.Bold))
        self.model_selector.setFont(QFont("Segoe UI", self._scaled(20, 12), QFont.Bold))
        self.avg_label.setFont(QFont("Segoe UI", self._scaled(18, 10), QFont.Bold))
        self.avg_label.setMinimumWidth(self._scaled(210, 130))
        self.status_label.setFont(QFont("Segoe UI", self._scaled(34, 18), QFont.Black))
        self.status_label.setMinimumHeight(self._scaled(112, 58))
        self.visualizer.setMinimumSize(self._scaled(280, 240), self._scaled(320, 260))
        self.selector_label.setVisible(self.width() >= 760)

        self._apply_avg_style(self._avg_color)
        self._apply_status_style(self._status_color)

    def _apply_avg_style(self, avg_color):
        self._avg_color = QColor(avg_color)
        light_color = self._avg_color.lighter(150)
        self.avg_label.setStyleSheet(
            f"""
            QLabel#AverageLabel {{
                background-color: rgba({self._avg_color.red()}, {self._avg_color.green()}, {self._avg_color.blue()}, 30);
                border: 1px solid rgba({self._avg_color.red()}, {self._avg_color.green()}, {self._avg_color.blue()}, 150);
                border-radius: {self._scaled(18, 9)}px;
                color: rgb({light_color.red()}, {light_color.green()}, {light_color.blue()});
                padding: {self._scaled(16, 8)}px {self._scaled(22, 10)}px;
            }}
            """
        )

    def _apply_status_style(self, label_color):
        self._status_color = QColor(label_color)
        self.status_label.setStyleSheet(
            f"""
            QLabel#StatusLabel {{
                background-color: rgba({self._status_color.red()}, {self._status_color.green()}, {self._status_color.blue()}, 34);
                border: 2px solid rgba({self._status_color.red()}, {self._status_color.green()}, {self._status_color.blue()}, 170);
                border-radius: {self._scaled(28, 12)}px;
                padding: {self._scaled(22, 10)}px;
                color: rgb({self._status_color.red()}, {self._status_color.green()}, {self._status_color.blue()});
            }}
            """
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_responsive_sizing()

    def load_model(self):
        selected_model = self.model_selector.currentText()
        if selected_model == "Simple Model":
            self.model = load_model(root_path("Training_Data", "models", "0_simple_cnn_model.keras"))
            self.scaler = joblib.load(root_path("Training_Data", "models", "0_simple_scaler.pkl"))
            print("Caricato modello semplice")
        elif selected_model == "LSTM":
            self.model = load_model(root_path("Training_Data", "models", "1_lstm_model.keras"))
            self.scaler = joblib.load(root_path("Training_Data", "models", "1_lstm_scaler.pkl"))
            self.realtime_window = [] 
            self.time_idx_counter = 0
            print("Caricato modello LSTM")
        elif selected_model == "Transformers":
            self.model = load_model(root_path("Training_Data", "models", "2_transformer_model.keras"))
            self.scaler = joblib.load(root_path("Training_Data", "models", "2_transformers_scaler.pkl"))
            self.realtime_window = [] 
            self.time_idx_counter = 0
            print("Caricato modello Transformers")
        elif selected_model == "ResNet1D":
            # Carica il modello completo salvato come file
            self.model = torch.load(root_path("Training_Data", "models", "3_resnet_full_model.pth"), weights_only=False)
            self.model.eval()  
            self.scaler = joblib.load(root_path("Training_Data", "models", "3_resnet_scaler.pkl"))
            print("Caricato modello ResNet1D")
        elif selected_model == "ResNet1D Sequence":
            # Carica il modello completo salvato come file
            self.model = torch.load(root_path("Training_Data", "models", "4_resnet_seq_full_model.pth"), weights_only=False)
            self.model.eval()  
            self.scaler = joblib.load(root_path("Training_Data", "models", "4_resnet1d_sequence_scaler.pkl"))
            self.realtime_window = [] 
            self.time_idx_counter = 0
            print("Caricato modello ResNet1D Sequence")

    def update_data(self):
        telem = read_telemetry()
        graphics = read_graphics()
        label_pred = NEUTRAL_LABEL

        if not should_force_neutral(telem):
            selected_model = self.model_selector.currentText()
            if selected_model == "Simple Model":
                processed_data = preprocess_realtime_data(telem, graphics, self.scaler, self.model_selector, self.time_idx_counter)
            elif selected_model == "LSTM" or selected_model == "Transformers":
                processed_data = preprocess_realtime_data_lstm(telem, graphics, self.scaler, self.realtime_window, self.window_size, self.model_selector, self.time_idx_counter)
            elif selected_model == "ResNet1D":
                processed_data = preprocess_realtime_data(telem, graphics, self.scaler, self.model_selector, self.time_idx_counter)
            elif selected_model == "ResNet1D Sequence":
                processed_data = preprocess_realtime_data_resnet_sequence(telem, graphics, self.scaler, self.realtime_window, self.window_size, self.model_selector, self.time_idx_counter)

            self.time_idx_counter += 1

            if processed_data is not None:
                if "ResNet" in selected_model:
                    processed_data = torch.tensor(processed_data, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
                    result = self.model(processed_data).argmax(dim=1).item()
                else:
                    result = predict_realtime(self.model, processed_data)
                label_pred = result


        for i in range(4):
            self.slip_window[i].append(telem["wheel_slip"][i])
            if len(self.slip_window[i]) > 8:
                self.slip_window[i].pop(0)

        slips = [sum(self.slip_window[i]) / len(self.slip_window[i]) for i in range(4)]

        self.visualizer.update_slips(slips)

        avg_slip = sum(slips) / len(slips)
        avg_color = interpolate_color(avg_slip)
        self.avg_label.setText(f"AVG SLIP {avg_slip:.2f}")
        self._apply_avg_style(avg_color)

        label_text, label_color = LABEL_STATES[label_pred]
        self.status_label.setText(f"{label_text.upper()}")
        self._apply_status_style(label_color)

        self.current_label_pred = label_pred
        self.sounds[label_pred].play()
        asyncio.run(send_data(label_pred, slips))

def convert_to_milliseconds(time_str: str) -> int:
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Formato non valido. Deve essere 'minuti:secondi:millisecondi'")
    
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])

    total_milliseconds = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
    return total_milliseconds

def preprocess_realtime_data(telem, graphics, scaler, model_selector, time_idx_counter):
    data = {
        "gas": telem["gas"],
        "brake": telem["brake"],
        "rpm": telem["rpm"],
        "steer": telem["steer"],
        "speed": telem["speed"],
        "g_force_x": telem["g_force"][2],
        "g_force_y": telem["g_force"][0], 
        "g_force_z": telem["g_force"][1],  
        "pressure_front_left": telem["pressure"][0],
        "pressure_front_right": telem["pressure"][1],
        "pressure_rear_left": telem["pressure"][2],
        "pressure_rear_right": telem["pressure"][3],
        "tyre_temp_front_left": telem["tyre_temp"][0],
        "tyre_temp_front_right": telem["tyre_temp"][1],
        "tyre_temp_rear_left": telem["tyre_temp"][2],
        "tyre_temp_rear_right": telem["tyre_temp"][3],
        "air_temp": telem["air_temp"],
        "road_temp": telem["road_temp"],
        "yaw_rate": telem["yaw_rate"],
        "normalized_car_position": graphics["normalized_car_position"],
        "wind_speed": graphics["wind_speed"],
        "wind_direction": graphics["wind_direction"],
        "current_time": convert_to_milliseconds(graphics["current_time_str"].rstrip('\x00')),
    }

    if model_selector.currentText() != "Simple Model":
        data["time_idx"] = time_idx_counter

    df = pd.DataFrame([data])
    scaled_features = scaler.transform(df)
    return scaled_features

def preprocess_realtime_data_lstm(telem, graphics, scaler, realtime_window, window_size, model_selector, time_idx_counter):
    processed_data = preprocess_realtime_data(telem, graphics, scaler, model_selector, time_idx_counter)
    realtime_window.append(processed_data[0])
    if len(realtime_window) < window_size:
        return None  

    realtime_window = realtime_window[-window_size:]
    return np.array([realtime_window])

def preprocess_realtime_data_resnet_sequence(telem, graphics, scaler, realtime_window, window_size, model_selector, time_idx_counter):
    processed_data = preprocess_realtime_data(telem, graphics, scaler, model_selector, time_idx_counter)
    realtime_window.append(processed_data[0])
    if len(realtime_window) < window_size:
        return None 

    realtime_window = realtime_window[-window_size:]
    return np.array([realtime_window])


def predict_realtime(model, processed_data):
    prediction = model.predict(processed_data, verbose=1)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

if __name__ == "__main__":
    threading.Thread(target=start_websocket_server, daemon=True).start()
    
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
