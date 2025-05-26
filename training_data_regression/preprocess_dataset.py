import pandas as pd
import os
import glob

def convert_to_milliseconds(time_str: str) -> int:
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Formato non valido. Deve essere 'minuti:secondi:millisecondi'")
    
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])

    total_milliseconds = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
    return total_milliseconds

def fix_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["current_time"] = df["current_time_str"].apply(convert_to_milliseconds)
    df = df[df["speed"] >= 0]

    df["wheel_slip_front_left"] = df["wheel_slip_front_left"].clip(upper=5)
    df["wheel_slip_front_right"] = df["wheel_slip_front_right"].clip(upper=5)
    df["wheel_slip_rear_left"] = df["wheel_slip_rear_left"].clip(upper=5)
    df["wheel_slip_rear_right"] = df["wheel_slip_rear_right"].clip(upper=5)

    df["avg_wheel_slip_front"] = (df["wheel_slip_front_left"] + df["wheel_slip_front_right"]) / 2
    df["avg_wheel_slip_rear"] = (df["wheel_slip_rear_left"] + df["wheel_slip_rear_right"]) / 2

    cols_to_drop = [
    'result',
    'wheel_slip_front_left', 'wheel_slip_front_right',
    'wheel_slip_rear_left', 'wheel_slip_rear_right',
    'current_time_str', 
    # "normalized_car_position","wind_speed","wind_direction",
    # "air_temp","road_temp",
    # "gas","brake","rpm","steer","speed","g_force_z","pressure_front_left","pressure_front_right","pressure_rear_left","pressure_rear_right",
    # "tyre_temp_front_left","tyre_temp_front_right","tyre_temp_rear_left","tyre_temp_rear_right","yaw_rate",
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    return df

def parse_filename(filename):
    base = os.path.basename(filename)
    parts = base.replace('.csv', '').split('_')
    driver = parts[2]
    track = parts[3]
    temp = parts[4]
    return driver, track, temp

def load_telemetry_data(folder_pattern="*.csv"):
    all_files = glob.glob(folder_pattern)
    rows = []

    for f in all_files:
        driver, track, temp = parse_filename(f)
        df = pd.read_csv(f)
        df['driver'] = driver
        df['track'] = track
        df['temp'] = temp
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

