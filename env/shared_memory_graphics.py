import re

from .ac_shared_memory import read_graphics_page


_TIME_PATTERN = re.compile(r"^\d+:\d{2}:\d{3}$")


def _clean_wchar(value, fallback=""):
    text = str(value).split("\x00", 1)[0].strip()
    return text or fallback


def _current_time(value):
    text = _clean_wchar(value, "0:00:000")
    if _TIME_PATTERN.match(text):
        return text
    return "0:00:000"


def read_graphics():
    page = read_graphics_page()
    if page is None:
        return {
            "current_time_str": "0:00:000",
            "normalized_car_position": 0.0,
            "wind_speed": 0.0,
            "wind_direction": 0.0,
        }

    wind_speed = float(page.windSpeed)
    wind_direction = 0.0 if abs(wind_speed) < 1e-6 else float(page.windDirection)

    return {
        "current_time_str": _current_time(page.currentTime),
        "normalized_car_position": float(page.normalizedCarPosition),
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
    }
