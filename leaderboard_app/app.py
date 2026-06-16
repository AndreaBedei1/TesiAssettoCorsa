import json
import os
import re
import sys
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(APP_DIR, "leaderboard_data.json")

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
SCREEN_FILL = 0.94
MIN_WINDOW_WIDTH = 540
MIN_WINDOW_HEIGHT = 700

TOP_LIMIT = 10

COLON_DOT_TIME = re.compile(r"^(\d+):([0-5]?\d)(?:[.,](\d{1,3}))?$")
COLON_COLON_TIME = re.compile(r"^(\d+):([0-5]?\d):(\d{1,3})$")


def app_path(*parts):
    return os.path.join(APP_DIR, *parts)


def normalize_name(name):
    return " ".join(name.strip().split())


def parse_lap_time(text):
    raw = text.strip().replace(",", ".")
    if not raw:
        raise ValueError("Inserisci un tempo.")

    colon_colon_match = COLON_COLON_TIME.match(raw)
    if colon_colon_match:
        minutes = int(colon_colon_match.group(1))
        seconds = int(colon_colon_match.group(2))
        milliseconds = int(colon_colon_match.group(3).ljust(3, "0")[:3])
        return ((minutes * 60) + seconds) * 1000 + milliseconds

    colon_dot_match = COLON_DOT_TIME.match(raw)
    if colon_dot_match:
        minutes = int(colon_dot_match.group(1))
        seconds = int(colon_dot_match.group(2))
        milliseconds = int((colon_dot_match.group(3) or "0").ljust(3, "0")[:3])
        return ((minutes * 60) + seconds) * 1000 + milliseconds

    try:
        seconds_value = float(raw)
    except ValueError as exc:
        raise ValueError("Formato tempo non valido. Usa 1:42.315 oppure 102.315.") from exc

    if seconds_value <= 0:
        raise ValueError("Il tempo deve essere maggiore di zero.")

    return int(round(seconds_value * 1000))


def format_lap_time(milliseconds):
    minutes, remainder = divmod(int(milliseconds), 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def format_delta(milliseconds, best_milliseconds):
    if best_milliseconds is None:
        return ""
    delta = int(milliseconds) - int(best_milliseconds)
    if delta <= 0:
        return "BEST"
    return f"+{delta / 1000:.3f}"


def load_entries():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    entries = []
    for item in data:
        name = normalize_name(str(item.get("name", "")))
        time_ms = item.get("time_ms")
        if not name or not isinstance(time_ms, int):
            continue
        entries.append(
            {
                "name": name,
                "time_ms": time_ms,
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", "")),
            }
        )
    return sorted(entries, key=lambda entry: entry["time_ms"])


def save_entries(entries):
    with open(DATA_FILE, "w", encoding="utf-8") as handle:
        json.dump(sorted(entries, key=lambda entry: entry["time_ms"]), handle, indent=2)


class LeaderboardStore:
    def __init__(self):
        self.entries = load_entries()

    def top_entries(self):
        return sorted(self.entries, key=lambda entry: entry["time_ms"])[:TOP_LIMIT]

    def best_time(self):
        top = self.top_entries()
        return top[0]["time_ms"] if top else None

    def add_or_update(self, name, time_ms):
        clean_name = normalize_name(name)
        if not clean_name:
            raise ValueError("Inserisci il nome del pilota.")
        if time_ms <= 0:
            raise ValueError("Il tempo deve essere maggiore di zero.")

        now = datetime.now().isoformat(timespec="seconds")
        existing = next(
            (entry for entry in self.entries if entry["name"].casefold() == clean_name.casefold()),
            None,
        )

        if existing is not None:
            if time_ms >= existing["time_ms"]:
                return False, f"{existing['name']} resta a {format_lap_time(existing['time_ms'])}."
            existing["name"] = clean_name
            existing["time_ms"] = time_ms
            existing["updated_at"] = now
            save_entries(self.entries)
            return True, f"{clean_name} migliora: {format_lap_time(time_ms)}."

        self.entries.append(
            {
                "name": clean_name,
                "time_ms": time_ms,
                "created_at": now,
                "updated_at": now,
            }
        )
        save_entries(self.entries)
        return True, f"{clean_name} entra in classifica: {format_lap_time(time_ms)}."

    def remove_driver(self, name):
        clean_name = normalize_name(name)
        before = len(self.entries)
        self.entries = [
            entry for entry in self.entries if entry["name"].casefold() != clean_name.casefold()
        ]
        if len(self.entries) == before:
            return False
        save_entries(self.entries)
        return True

    def reset(self):
        self.entries = []
        save_entries(self.entries)


class LeaderRow(QFrame):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.setObjectName("LeaderRow")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(22, 14, 22, 14)
        self.layout.setSpacing(18)

        self.rank_label = QLabel(f"{rank:02d}")
        self.rank_label.setAlignment(Qt.AlignCenter)
        self.rank_label.setObjectName("RankLabel")

        self.name_label = QLabel("Slot libero")
        self.name_label.setObjectName("DriverLabel")
        self.name_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.time_label = QLabel("--:--.---")
        self.time_label.setObjectName("TimeLabel")
        self.time_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.delta_label = QLabel("")
        self.delta_label.setObjectName("DeltaLabel")
        self.delta_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.rank_label)
        self.layout.addWidget(self.name_label, stretch=1)
        self.layout.addWidget(self.time_label)
        self.layout.addWidget(self.delta_label)

    def update_content(self, entry, best_time_ms, scale):
        height = max(48, int(round(92 * scale)))
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)

        self.layout.setContentsMargins(
            max(10, int(round(22 * scale))),
            max(6, int(round(14 * scale))),
            max(10, int(round(22 * scale))),
            max(6, int(round(14 * scale))),
        )
        self.layout.setSpacing(max(8, int(round(18 * scale))))

        self.rank_label.setFixedWidth(max(46, int(round(72 * scale))))
        self.delta_label.setFixedWidth(max(68, int(round(112 * scale))))
        self.rank_label.setFont(QFont("Segoe UI", max(14, int(round(26 * scale))), QFont.Black))
        self.name_label.setFont(QFont("Segoe UI", max(14, int(round(25 * scale))), QFont.Bold))
        self.time_label.setFont(QFont("Consolas", max(15, int(round(30 * scale))), QFont.Bold))
        self.delta_label.setFont(QFont("Segoe UI", max(11, int(round(17 * scale))), QFont.Bold))

        if entry is None:
            self.name_label.setText("Slot libero")
            self.time_label.setText("--:--.---")
            self.delta_label.setText("")
            self._apply_style("#111820", "#26313c", "#667085", "#cbd5e1", "#111820", muted=True)
            return

        self.name_label.setText(entry["name"].upper())
        self.time_label.setText(format_lap_time(entry["time_ms"]))
        self.delta_label.setText(format_delta(entry["time_ms"], best_time_ms))

        if self.rank == 1:
            self._apply_style("#1b1608", "#f6c453", "#f6c453", "#fff7cc", "#2a210b")
        elif self.rank == 2:
            self._apply_style("#14191f", "#b8c2cc", "#d8dee6", "#f8fafc", "#1f2730")
        elif self.rank == 3:
            self._apply_style("#1d120c", "#c9783d", "#f2a66f", "#fff1df", "#2a1a11")
        else:
            self._apply_style("#101821", "#2a3a48", "#5dd6ff", "#f8fafc", "#0f222b")

    def _apply_style(self, background, border, accent, text, badge, muted=False):
        muted_text = "#64748b" if muted else text
        self.setStyleSheet(
            f"""
            QFrame#LeaderRow {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 18px;
            }}
            QLabel#RankLabel {{
                background-color: {badge};
                border: 1px solid {accent};
                border-radius: 13px;
                color: {accent};
                padding: 4px;
            }}
            QLabel#DriverLabel {{
                background-color: transparent;
                color: {muted_text};
            }}
            QLabel#TimeLabel {{
                background-color: transparent;
                color: {muted_text};
            }}
            QLabel#DeltaLabel {{
                background-color: rgba(255, 255, 255, 18);
                border-radius: 12px;
                color: {accent};
                padding: 5px;
            }}
            """
        )


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.store = LeaderboardStore()
        self._ui_scale = 1.0

        self.setWindowTitle("Time Attack Leaderboard")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(20)

        self.header_panel = QFrame()
        self.header_panel.setObjectName("HeaderPanel")
        self.header_layout = QVBoxLayout(self.header_panel)
        self.header_layout.setContentsMargins(34, 30, 34, 30)
        self.header_layout.setSpacing(8)

        self.title_label = QLabel("TIME ATTACK")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 58, QFont.Black))

        self.subtitle_label = QLabel("CLASSIFICA LIVE - TOP 10")
        self.subtitle_label.setObjectName("SubtitleLabel")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setFont(QFont("Segoe UI", 18, QFont.Bold))

        self.header_layout.addWidget(self.title_label)
        self.header_layout.addWidget(self.subtitle_label)

        self.metrics_grid = QGridLayout()
        self.metrics_grid.setSpacing(16)
        self.best_card, self.best_value = self._create_metric_card("BEST LAP", "--:--.---")
        self.count_card, self.count_value = self._create_metric_card("PILOTI", "0")
        self.cut_card, self.cut_value = self._create_metric_card("TOP 10 CUT", "--:--.---")
        self.metrics_grid.addWidget(self.best_card, 0, 0)
        self.metrics_grid.addWidget(self.count_card, 0, 1)
        self.metrics_grid.addWidget(self.cut_card, 0, 2)

        self.input_panel = QFrame()
        self.input_panel.setObjectName("InputPanel")
        self.input_layout = QGridLayout(self.input_panel)
        self.input_layout.setContentsMargins(26, 24, 26, 24)
        self.input_layout.setHorizontalSpacing(14)
        self.input_layout.setVerticalSpacing(12)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nome pilota")
        self.name_input.setObjectName("InputField")

        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("Tempo 1:42.315")
        self.time_input.setObjectName("InputField")

        self.add_button = QPushButton("AGGIUNGI")
        self.add_button.setObjectName("PrimaryButton")
        self.remove_button = QPushButton("RIMUOVI")
        self.remove_button.setObjectName("SecondaryButton")
        self.reset_button = QPushButton("RESET")
        self.reset_button.setObjectName("GhostButton")

        self.name_input.returnPressed.connect(self.add_time)
        self.time_input.returnPressed.connect(self.add_time)
        self.add_button.clicked.connect(self.add_time)
        self.remove_button.clicked.connect(self.remove_driver)
        self.reset_button.clicked.connect(self.confirm_reset)

        self.input_layout.addWidget(self.name_input, 0, 0)
        self.input_layout.addWidget(self.time_input, 0, 1)
        self.input_layout.addWidget(self.add_button, 0, 2)
        self.input_layout.addWidget(self.remove_button, 1, 1)
        self.input_layout.addWidget(self.reset_button, 1, 2)

        self.message_label = QLabel("Pronto per una nuova sfida.")
        self.message_label.setObjectName("MessageLabel")
        self.message_label.setAlignment(Qt.AlignCenter)

        self.board_panel = QFrame()
        self.board_panel.setObjectName("BoardPanel")
        self.board_layout = QVBoxLayout(self.board_panel)
        self.board_layout.setContentsMargins(24, 24, 24, 24)
        self.board_layout.setSpacing(12)

        self.rows = []
        for rank in range(1, TOP_LIMIT + 1):
            row = LeaderRow(rank)
            self.rows.append(row)
            self.board_layout.addWidget(row)

        self.main_layout.addWidget(self.header_panel)
        self.main_layout.addLayout(self.metrics_grid)
        self.main_layout.addWidget(self.input_panel)
        self.main_layout.addWidget(self.message_label)
        self.main_layout.addWidget(self.board_panel, stretch=1)

        self.fit_to_available_screen()
        self.apply_responsive_sizing()
        self.refresh_board()

    def _create_metric_card(self, label, value):
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(4)

        label_widget = QLabel(label)
        label_widget.setObjectName("MetricLabel")
        label_widget.setAlignment(Qt.AlignCenter)

        value_widget = QLabel(value)
        value_widget.setObjectName("MetricValue")
        value_widget.setAlignment(Qt.AlignCenter)

        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        return card, value_widget

    def _scaled(self, value, minimum=1):
        return max(minimum, int(round(value * self._ui_scale)))

    def _stylesheet(self):
        return f"""
            QWidget {{
                background-color: #06080c;
                color: #f8fafc;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}

            QFrame#HeaderPanel,
            QFrame#InputPanel,
            QFrame#BoardPanel,
            QFrame#MetricCard {{
                background-color: #101720;
                border: 1px solid #26323e;
                border-radius: {self._scaled(24, 10)}px;
            }}

            QLabel#TitleLabel {{
                background-color: transparent;
                color: #ffffff;
            }}

            QLabel#SubtitleLabel {{
                background-color: transparent;
                color: #5dd6ff;
            }}

            QLabel#MetricLabel {{
                background-color: transparent;
                color: #7b8da1;
            }}

            QLabel#MetricValue {{
                background-color: transparent;
                color: #f6c453;
            }}

            QLabel#MessageLabel {{
                background-color: rgba(93, 214, 255, 20);
                border: 1px solid rgba(93, 214, 255, 90);
                border-radius: {self._scaled(16, 8)}px;
                color: #dff7ff;
                padding: {self._scaled(12, 6)}px;
            }}

            QLineEdit#InputField {{
                background-color: #0b1118;
                border: 1px solid #334252;
                border-radius: {self._scaled(15, 8)}px;
                color: #f8fafc;
                padding: {self._scaled(16, 8)}px {self._scaled(18, 10)}px;
                selection-background-color: #1b6f8c;
            }}

            QLineEdit#InputField:focus {{
                border: 2px solid #5dd6ff;
            }}

            QPushButton {{
                border-radius: {self._scaled(15, 8)}px;
                padding: {self._scaled(15, 8)}px {self._scaled(18, 10)}px;
                font-weight: 800;
            }}

            QPushButton#PrimaryButton {{
                background-color: #f6c453;
                border: 1px solid #ffe28a;
                color: #101014;
            }}

            QPushButton#PrimaryButton:hover {{
                background-color: #ffda74;
            }}

            QPushButton#SecondaryButton {{
                background-color: #102532;
                border: 1px solid #2d8bad;
                color: #dff7ff;
            }}

            QPushButton#SecondaryButton:hover {{
                background-color: #16384a;
            }}

            QPushButton#GhostButton {{
                background-color: #171d25;
                border: 1px solid #394756;
                color: #9fb0c2;
            }}

            QPushButton#GhostButton:hover {{
                background-color: #202935;
                color: #f8fafc;
            }}
        """

    def fit_to_available_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(TARGET_WIDTH, TARGET_HEIGHT)
            return

        available = screen.availableGeometry()
        max_w = max(MIN_WINDOW_WIDTH, int(available.width() * SCREEN_FILL))
        max_h = max(MIN_WINDOW_HEIGHT, int(available.height() * SCREEN_FILL))

        frame_extra_w = max(16, self.frameGeometry().width() - self.geometry().width())
        frame_extra_h = max(48, self.frameGeometry().height() - self.geometry().height())
        target_w = max(MIN_WINDOW_WIDTH, min(TARGET_WIDTH, max_w - frame_extra_w))
        target_h = max(MIN_WINDOW_HEIGHT, min(TARGET_HEIGHT, max_h - frame_extra_h))
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
        if not hasattr(self, "rows"):
            return

        self._ui_scale = max(
            0.52,
            min(1.0, min(self.width() / TARGET_WIDTH, self.height() / TARGET_HEIGHT)),
        )
        self.setStyleSheet(self._stylesheet())

        outer = self._scaled(40, 18)
        self.main_layout.setContentsMargins(outer, outer, outer, outer)
        self.main_layout.setSpacing(self._scaled(20, 8))

        self.header_layout.setContentsMargins(
            self._scaled(34, 14),
            self._scaled(30, 12),
            self._scaled(34, 14),
            self._scaled(30, 12),
        )
        self.header_layout.setSpacing(self._scaled(8, 3))

        self.input_layout.setContentsMargins(
            self._scaled(26, 12),
            self._scaled(24, 10),
            self._scaled(26, 12),
            self._scaled(24, 10),
        )
        self.input_layout.setHorizontalSpacing(self._scaled(14, 7))
        self.input_layout.setVerticalSpacing(self._scaled(12, 6))
        self.board_layout.setContentsMargins(
            self._scaled(24, 10),
            self._scaled(24, 10),
            self._scaled(24, 10),
            self._scaled(24, 10),
        )
        self.board_layout.setSpacing(self._scaled(12, 5))
        self.metrics_grid.setSpacing(self._scaled(16, 6))

        self.title_label.setFont(QFont("Segoe UI", self._scaled(58, 26), QFont.Black))
        self.subtitle_label.setFont(QFont("Segoe UI", self._scaled(18, 10), QFont.Bold))
        self.message_label.setFont(QFont("Segoe UI", self._scaled(16, 9), QFont.Bold))
        self.name_input.setFont(QFont("Segoe UI", self._scaled(18, 10), QFont.Bold))
        self.time_input.setFont(QFont("Consolas", self._scaled(19, 11), QFont.Bold))
        for button in (self.add_button, self.remove_button, self.reset_button):
            button.setFont(QFont("Segoe UI", self._scaled(16, 9), QFont.Black))

        for card in (self.best_card, self.count_card, self.cut_card):
            card.layout().setContentsMargins(
                self._scaled(20, 8),
                self._scaled(18, 8),
                self._scaled(20, 8),
                self._scaled(18, 8),
            )
            card.layout().setSpacing(self._scaled(4, 1))

        for value_label in (self.best_value, self.count_value, self.cut_value):
            value_label.setFont(QFont("Consolas", self._scaled(28, 14), QFont.Bold))
        for card in (self.best_card, self.count_card, self.cut_card):
            card.findChildren(QLabel)[0].setFont(
                QFont("Segoe UI", self._scaled(13, 8), QFont.Bold)
            )

        self.refresh_board()

    def refresh_board(self):
        top_entries = self.store.top_entries()
        best = self.store.best_time()

        for index, row in enumerate(self.rows):
            entry = top_entries[index] if index < len(top_entries) else None
            row.update_content(entry, best, self._ui_scale)

        self.best_value.setText(format_lap_time(best) if best is not None else "--:--.---")
        self.count_value.setText(str(len(self.store.entries)))
        if len(top_entries) >= TOP_LIMIT:
            self.cut_value.setText(format_lap_time(top_entries[-1]["time_ms"]))
        else:
            self.cut_value.setText("--:--.---")

    def add_time(self):
        name = self.name_input.text()
        try:
            time_ms = parse_lap_time(self.time_input.text())
            changed, message = self.store.add_or_update(name, time_ms)
        except ValueError as exc:
            self.show_message(str(exc), error=True)
            return

        self.show_message(message, error=not changed)
        if changed:
            self.name_input.clear()
            self.time_input.clear()
        self.refresh_board()

    def remove_driver(self):
        name = normalize_name(self.name_input.text())
        if not name:
            self.show_message("Scrivi il nome del pilota da rimuovere.", error=True)
            return

        if self.store.remove_driver(name):
            self.show_message(f"{name} rimosso dalla classifica.")
            self.name_input.clear()
            self.refresh_board()
        else:
            self.show_message(f"{name} non trovato.", error=True)

    def confirm_reset(self):
        if not self.store.entries:
            self.show_message("La classifica e gia vuota.")
            return

        result = QMessageBox.question(
            self,
            "Reset classifica",
            "Vuoi cancellare tutti i tempi salvati?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            return

        self.store.reset()
        self.show_message("Classifica azzerata.")
        self.refresh_board()

    def show_message(self, text, error=False):
        color = "#ff5c6c" if error else "#5dd6ff"
        self.message_label.setText(text)
        self.message_label.setStyleSheet(
            f"""
            QLabel#MessageLabel {{
                background-color: rgba({255 if error else 93}, {92 if error else 214}, {108 if error else 255}, 20);
                border: 1px solid {color};
                border-radius: {self._scaled(16, 8)}px;
                color: {color};
                padding: {self._scaled(12, 6)}px;
            }}
            """
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_responsive_sizing()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
