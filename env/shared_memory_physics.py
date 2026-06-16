from .ac_shared_memory import read_physics_page


def _float_list(values):
    return [float(value) for value in values]


def _default_telemetry():
    return {
        "gas": 0.0,
        "brake": 0.0,
        "rpm": 0,
        "steer": 0.0,
        "speed": 0.0,
        "g_force": [0.0, 0.0, 0.0],
        "wheel_slip": [0.0, 0.0, 0.0, 0.0],
        "pressure": [0.0, 0.0, 0.0, 0.0],
        "tyre_temp": [0.0, 0.0, 0.0, 0.0],
        "air_temp": 0.0,
        "road_temp": 0.0,
        "yaw_rate": 0.0,
    }


def read_telemetry():
    page = read_physics_page()
    if page is None:
        return _default_telemetry()

    # Assetto Corsa uses Y as the vertical axis, so yaw is the angular velocity around Y.
    yaw_rate = float(page.localAngularVel[1])

    return {
        "gas": float(page.gas),
        "brake": float(page.brake),
        "rpm": int(page.rpms),
        "steer": float(page.steerAngle),
        "speed": float(page.speedKmh),
        "g_force": _float_list(page.accG),
        "wheel_slip": _float_list(page.wheelSlip),
        "pressure": _float_list(page.wheelsPressure),
        "tyre_temp": _float_list(page.tyreCoreTemperature),
        "air_temp": float(page.airTemp),
        "road_temp": float(page.roadTemp),
        "yaw_rate": yaw_rate,
    }

