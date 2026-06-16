import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from env.ac_shared_memory import read_physics_page
from env.shared_memory_graphics import read_graphics
from env.shared_memory_physics import read_telemetry
from utils.classify_vehicle_behavior import classify_vehicle_state


FIELDS = [
    "gas",
    "brake",
    "rpm",
    "steer",
    "speed",
    "g_force_x",
    "g_force_y",
    "g_force_z",
    "wheel_slip_front_left",
    "wheel_slip_front_right",
    "wheel_slip_rear_left",
    "wheel_slip_rear_right",
    "pressure_front_left",
    "pressure_front_right",
    "pressure_rear_left",
    "pressure_rear_right",
    "tyre_temp_front_left",
    "tyre_temp_front_right",
    "tyre_temp_rear_left",
    "tyre_temp_rear_right",
    "air_temp",
    "road_temp",
    "yaw_rate",
    "current_time_str",
    "normalized_car_position",
    "wind_speed",
    "wind_direction",
    "result",
]


def default_output_path(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vehicle_telemetry_{args.driver}_{args.track}_{args.temp}_{timestamp}.csv"
    return ROOT_DIR / "data" / "new_recordings" / filename


def telemetry_row(telem, graphics):
    result = classify_vehicle_state(telem["speed"], telem["g_force"], telem["wheel_slip"])
    return {
        "gas": telem["gas"],
        "brake": telem["brake"],
        "rpm": telem["rpm"],
        "steer": telem["steer"],
        "speed": telem["speed"],
        "g_force_x": telem["g_force"][2],
        "g_force_y": telem["g_force"][0],
        "g_force_z": telem["g_force"][1],
        "wheel_slip_front_left": telem["wheel_slip"][0],
        "wheel_slip_front_right": telem["wheel_slip"][1],
        "wheel_slip_rear_left": telem["wheel_slip"][2],
        "wheel_slip_rear_right": telem["wheel_slip"][3],
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
        "current_time_str": graphics["current_time_str"].rstrip("\x00"),
        "normalized_car_position": graphics["normalized_car_position"],
        "wind_speed": graphics["wind_speed"],
        "wind_direction": graphics["wind_direction"],
        "result": result,
    }


def wait_for_shared_memory():
    while read_physics_page() is None:
        print("Aspetto Assetto Corsa shared memory...", flush=True)
        time.sleep(1.0)


def wait_until_moving(min_speed):
    if min_speed <= 0:
        return

    while True:
        telem = read_telemetry()
        if telem["speed"] >= min_speed:
            return
        print(f"Aspetto che la macchina si muova: {telem['speed']:.1f}/{min_speed:.1f} km/h", flush=True)
        time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="Registra telemetria Assetto Corsa dal nuovo env shared-memory.")
    parser.add_argument("--output", type=Path, default=None, help="CSV di destinazione.")
    parser.add_argument("--duration", type=float, default=180.0, help="Durata in secondi. Usa 0 per fermare con Ctrl+C.")
    parser.add_argument("--samples", type=int, default=None, help="Numero massimo di righe da registrare.")
    parser.add_argument("--hz", type=float, default=10.0, help="Frequenza di campionamento.")
    parser.add_argument("--driver", default="andrea_live", help="Nome pilota nel filename.")
    parser.add_argument("--track", default="unknown", help="Pista nel filename.")
    parser.add_argument("--temp", default="unknown", help="Temperatura/setup nel filename, es. 36G.")
    parser.add_argument("--start-when-speed", type=float, default=5.0, help="Parte quando speed supera questa soglia.")
    parser.add_argument("--no-wait", action="store_true", help="Non aspettare shared memory prima di partire.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("--hz deve essere maggiore di 0")
    if args.duration is not None and args.duration <= 0:
        args.duration = None

    output_path = args.output or default_output_path(args)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_wait:
        wait_for_shared_memory()
        wait_until_moving(args.start_when_speed)

    interval = 1.0 / args.hz
    started_at = time.perf_counter()
    rows_written = 0
    next_status_at = started_at

    print(f"Registro in: {output_path}", flush=True)
    print("Premi Ctrl+C per fermare.", flush=True)

    try:
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()

            while True:
                loop_started = time.perf_counter()
                elapsed = loop_started - started_at
                if args.duration is not None and elapsed >= args.duration:
                    break
                if args.samples is not None and rows_written >= args.samples:
                    break

                telem = read_telemetry()
                graphics = read_graphics()
                writer.writerow(telemetry_row(telem, graphics))
                rows_written += 1

                if loop_started >= next_status_at:
                    print(
                        f"{rows_written} righe | speed={telem['speed']:.1f} km/h "
                        f"steer={telem['steer']:.4f} yaw={telem['yaw_rate']:.4f}",
                        flush=True,
                    )
                    next_status_at = loop_started + 1.0

                sleep_for = interval - (time.perf_counter() - loop_started)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nRegistrazione fermata manualmente.", flush=True)

    print(f"Salvate {rows_written} righe in {output_path}", flush=True)


if __name__ == "__main__":
    main()
