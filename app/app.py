import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QComboBox
)
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QUrl
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.shared_memory_physics import read_telemetry
from env.shared_memory_graphics import read_graphics
from training_data.r1 import ResNet1DTabular
from training_data.r1s import ResNet1D

model = load_model("../Training_Data/models/0_simple_cnn_model.keras")
scaler = joblib.load("../Training_Data/models/0_simple_scaler.pkl")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training_data')))




# Lista dei client connessi
connected_clients = set()

async def websocket_handler(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        connected_clients.remove(websocket)

async def send_data(label_pred, slips):
    if connected_clients:
        data = {
            "label_pred": label_pred,
            "slips": slips
        }
        message = json.dumps(data)
        await asyncio.wait([client.send(message) for client in connected_clients])

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_server():
        server = await websockets.serve(websocket_handler, "localhost", 8765)
        print("WebSocket server avviato su ws://localhost:8765")
        await server.wait_closed()

    # Esegui la coroutine
    loop.run_until_complete(run_server())
    loop.run_forever()

LABEL_STATES = {
    0: ("Grip Loss", QColor(255, 0, 0)),
    1: ("High Grip - Accelerate", QColor(0, 255, 0)),
    2: ("Low Grip", QColor(255, 165, 0)),
    3: ("Neutral", QColor(0, 128, 0)),
}

def interpolate_color(value, max_val=1.0):
    ratio = min(max(value / max_val, 0.0), 1.0)
    hue = (120 - 120 * ratio) / 360.0  # green to red
    color = QColor.fromHsvF(hue, 0.85, 0.9)  # più desaturato, stile vintage
    return color

class WheelVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.slips = [0.0] * 4
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f3e8d3; border: 2px solid #a18860;")


    def update_slips(self, slips):
        self.slips = slips
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        car_w = w * 0.2
        car_h = h * 0.8
        axle_offset = h * 0.22
        wheel_r = min(w, h) * 0.10

        center_x = w // 2
        center_y = h // 2

        # Car body
        car_rect_x = center_x - car_w / 2
        car_rect_y = center_y - car_h / 2
        painter.setBrush(QColor(40, 40, 40))
        painter.setPen(QPen(QColor(100, 200, 255), 2)) 
        painter.drawRoundedRect(int(car_rect_x + 17), int(car_rect_y), int(car_w - 30), int(car_h), 10, 10)

        # Axles
        painter.setBrush(QColor(90, 90, 90))
        painter.setPen(QPen(QColor(100, 200, 255), 2)) 
        painter.drawRoundedRect(int(car_rect_x), int(center_y - axle_offset - 5), int(car_w), 10, 3, 3)
        painter.drawRoundedRect(int(car_rect_x), int(center_y + axle_offset - 5), int(car_w), 10, 3, 3)

        # Wheels
        wheel_positions = [
            (car_rect_x - wheel_r * 2, center_y - axle_offset - wheel_r),
            (car_rect_x + car_w , center_y - axle_offset - wheel_r),
            (car_rect_x - wheel_r * 2, center_y + axle_offset - wheel_r),
            (car_rect_x + car_w, center_y + axle_offset - wheel_r),
        ]

        font_size = max(8, int(wheel_r * 0.35))
        for i, (x, y) in enumerate(wheel_positions):
            slip = self.slips[i]
            color = interpolate_color(slip)

            painter.setBrush(color)
            painter.setPen(QPen(Qt.black, 1.2))
            painter.drawEllipse(int(x), int(y), int(wheel_r * 2), int(wheel_r * 2))

            painter.setPen(QPen(Qt.black))
            painter.setFont(QFont("Georgia", font_size, QFont.Bold))
            painter.drawText(
                int(x + wheel_r / 2), int(y + wheel_r * 2 + font_size + 10), f"{slip:.2f}"
            )

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slip Dashboard")
        self.setMinimumSize(600, 500)

        self.setWindowIcon(QIcon("img.png"))

        self.current_label_pred = None  # salva l'ultima label mostrata
        self.time_idx_counter = 0
        self.slip_window = [[], [], [], []]

        

        # Carica i suoni
        self.sounds = {
            0: QSoundEffect(),
            1: QSoundEffect(),
            2: QSoundEffect(),
            3: QSoundEffect()
        }
        self.sounds[1].setSource(QUrl.fromLocalFile("high_grip.wav"))
        self.sounds[2].setSource(QUrl.fromLocalFile("limit_grip.wav"))
        self.sounds[0].setSource(QUrl.fromLocalFile("loss_grip.wav"))

        for s in self.sounds.values():
            s.setVolume(0.5)

        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Roboto', sans-serif;
            }

            QComboBox {
                border: 2px solid #0078d7;
                border-radius: 10px;
                padding: 10px;
                background-color: #f0f4f8;
                font-size: 14px;
                color: #333333;
                padding-right: 30px; 
            }

            QComboBox:hover {
                background-color: #e6f0fa;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #0078d7;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }

            QComboBox::down-arrow {
                image: url(arrow.svg);  
                width: 25%;
                height: 25%;
                margin-right: 7px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #0078d7;
                selection-background-color: #0078d7;
                selection-color: white;
                padding: 5px;
                background-color: #ffffff;
                outline: none;
            }
        """)


        self.visualizer = WheelVisualizer()

        self.status_label = QLabel("Status: ---")
        self.status_label.setFont(QFont("Roboto", 16, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: #f0f4f8;
            border: 2px solid #0078d7;
            border-radius: 12px;
            padding: 16px;
            color: #333333;
        """)

        # Modello selezionabile
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Simple Model", "LSTM", "Transformers", "ResNet1D", "ResNet1D Sequence"])
        self.model_selector.setFont(QFont("Roboto", 14))
        self.model_selector.currentIndexChanged.connect(self.load_model)

        layout = QVBoxLayout()
        layout.addWidget(self.model_selector)
        layout.addWidget(self.visualizer, stretch=1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(150)

        self.model = None
        self.scaler = None
        self.window_size = 5
        self.realtime_window = []

        self.load_model()

    def load_model(self):
        selected_model = self.model_selector.currentText()
        if selected_model == "Simple Model":
            self.model = load_model("../Training_Data/models/0_simple_cnn_model.keras")
            self.scaler = joblib.load("../Training_Data/models/0_simple_scaler.pkl")
            print("Caricato modello semplice")
        elif selected_model == "LSTM":
            self.model = load_model("../Training_Data/models/1_lstm_model.keras")
            self.scaler = joblib.load("../Training_Data/models/1_lstm_scaler.pkl")
            self.realtime_window = []  # Reset della finestra temporale
            self.time_idx_counter = 0
            print("Caricato modello LSTM")
        elif selected_model == "Transformers":
            self.model = load_model("../Training_Data/models/2_transformer_model.keras")
            self.scaler = joblib.load("../Training_Data/models/2_transformers_scaler.pkl")
            self.realtime_window = []  # Reset della finestra temporale
            self.time_idx_counter = 0
            print("Caricato modello Transformers")
        elif selected_model == "ResNet1D":
            # Carica il modello completo salvato come file
            self.model = torch.load("../Training_Data/models/3_resnet_full_model.pth", weights_only=False)
            self.model.eval()  # Imposta il modello in modalità di inferenza
            self.scaler = joblib.load("../Training_Data/models/3_resnet_scaler.pkl")
            print("Caricato modello ResNet1D")
        elif selected_model == "ResNet1D Sequence":
            # Carica il modello completo salvato come file
            self.model = torch.load("../Training_Data/models/4_resnet_seq_full_model.pth", weights_only=False)
            self.model.eval()  # Imposta il modello in modalità di inferenza
            self.scaler = joblib.load("../Training_Data/models/4_resnet1d_sequence_scaler.pkl")
            self.realtime_window = []  # Reset della finestra temporale
            self.time_idx_counter = 0
            print("Caricato modello ResNet1D Sequence")

    def update_data(self):
        telem = read_telemetry()
        graphics = read_graphics()
        label_pred = 0

        label_pred = 3
        if telem["speed"] < 20:
            label_pred = 3
        else:
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

        label_text, label_color = LABEL_STATES[label_pred]
        self.status_label.setText(f"{label_text}")
        self.status_label.setStyleSheet(
            f"""
            background-color: #f0f4f8;
            border: 2px solid #0078d7;
            border-radius: 12px;
            padding: 16px;
            color: rgb({label_color.red()}, {label_color.green()}, {label_color.blue()});
            """
        )

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
        "g_force_x": telem["g_force"][2],  # Inversione a causa di un bug
        "g_force_y": telem["g_force"][0],  # Inversione a causa di un bug
        "g_force_z": telem["g_force"][1],  # Inversione a causa di un bug
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