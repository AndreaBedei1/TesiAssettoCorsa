import os
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl
import pandas as pd

mcap_files = [
    "../logall/rosbag-20250403T141242-08-control/rosbag-20250403T141242-08-control_0.mcap",
    "../logall/rosbag-20250403T141242-08-control/rosbag-20250403T141242-08-control_1.mcap"
]

idl_dir = os.path.join("..", "arp", "idl")

topic_configs = {
    "/observer/ego_loc": {
        "msg_type": "musa/msg/Localization",
        "csv_file": "ego_loc_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "position_x": lambda msg: msg.position.x,
            "position_y": lambda msg: msg.position.y,
            "position_z": lambda msg: msg.position.z,
            "orientation_ypr_x": lambda msg: msg.orientation_ypr.x,
            "orientation_ypr_y": lambda msg: msg.orientation_ypr.y,
            "orientation_ypr_z": lambda msg: msg.orientation_ypr.z,
        }
    },
    "/observer/ego_state": {
        "msg_type": "musa/msg/EgoState",
        "csv_file": "ego_state_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "velocity_x": lambda msg: msg.velocity.x,
            "velocity_y": lambda msg: msg.velocity.y,
            "velocity_z": lambda msg: msg.velocity.z,
            "angular_rate_x": lambda msg: msg.angular_rate.x,
            "angular_rate_y": lambda msg: msg.angular_rate.y,
            "angular_rate_z": lambda msg: msg.angular_rate.z,
            "acceleration_x": lambda msg: msg.acceleration.x,
            "acceleration_y": lambda msg: msg.acceleration.y,
            "acceleration_z": lambda msg: msg.acceleration.z,
            "wheels_speed_fl": lambda msg: msg.wheels_speed.fl,
            "wheels_speed_fr": lambda msg: msg.wheels_speed.fr,
            "wheels_speed_rl": lambda msg: msg.wheels_speed.rl,
            "wheels_speed_rr": lambda msg: msg.wheels_speed.rr,
            "wheels_toe_angle_fl": lambda msg: msg.wheels_toe_angle.fl,
            "wheels_toe_angle_fr": lambda msg: msg.wheels_toe_angle.fr,
            "wheels_toe_angle_rl": lambda msg: msg.wheels_toe_angle.rl,
            "wheels_toe_angle_rr": lambda msg: msg.wheels_toe_angle.rr,
        }
    },
    "/eav24_badenia/wheel_load": {
        "msg_type": "eav24_badenia/msg/WheelLoad",
        "csv_file": "wheel_load_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "load_wheel_fl": lambda msg: msg.load_wheel_fl,
            "load_wheel_fr": lambda msg: msg.load_wheel_fr,
            "load_wheel_rr": lambda msg: msg.load_wheel_rr,
            "load_wheel_rl": lambda msg: msg.load_wheel_rl,
        }
    },
    "/eav24_badenia/tpms_front": {
        "msg_type": "eav24_badenia/msg/TPMS_Front",
        "csv_file": "tpms_front_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "tpr4_temp_fl": lambda msg: msg.tpr4_temp_fl,
            "tpr4_temp_fr": lambda msg: msg.tpr4_temp_fr,
            "tpr4_abs_press_fl": lambda msg: msg.tpr4_abs_press_fl,
            "tpr4_abs_press_fr": lambda msg: msg.tpr4_abs_press_fr,
        }
    },
    "/eav24_badenia/tpms_rear": {
        "msg_type": "eav24_badenia/msg/TPMS_Rear",
        "csv_file": "tpms_rear_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "tpr4_temp_rl": lambda msg: msg.tpr4_temp_rl,
            "tpr4_temp_rr": lambda msg: msg.tpr4_temp_rr,
            "tpr4_abs_press_rl": lambda msg: msg.tpr4_abs_press_rl,
            "tpr4_abs_press_rr": lambda msg: msg.tpr4_abs_press_rr,
        }
    },
    "/eav24_badenia/ride_front": {
        "msg_type": "eav24_badenia/msg/Ride_Front",
        "csv_file": "ride_front_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "damper_stroke_fl": lambda msg: msg.damper_stroke_fl,
            "damper_stroke_fr": lambda msg: msg.damper_stroke_fr,
        }
    },
    "/eav24_badenia/ride_rear": {
        "msg_type": "eav24_badenia/msg/Ride_Rear",
        "csv_file": "ride_rear_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "damper_stroke_rl": lambda msg: msg.damper_stroke_rl,
            "damper_stroke_rr": lambda msg: msg.damper_stroke_rr,
        }
    },
    "/rmpc/debug": {
        "msg_type": "rmpc/msg/Debug",
        "csv_file": "rmpc_debug_data.csv",
        "fields": {
            "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
            "predicted_fzf": lambda msg: msg.predicted_fzf,
            "predicted_fzr": lambda msg: msg.predicted_fzr,
            "predicted_fyf": lambda msg: msg.predicted_fyf,
            "predicted_fyr": lambda msg: msg.predicted_fyr,
            "predicted_fxf": lambda msg: msg.predicted_fxf,
            "predicted_fxr": lambda msg: msg.predicted_fxr,
            "predicted_alpha_f": lambda msg: msg.predicted_alpha_f,
            "predicted_alpha_r": lambda msg: msg.predicted_alpha_r,
        }
    },
}

typestore = get_typestore(Stores.ROS2_HUMBLE)

def preprocess_idl(content):
    lines = content.splitlines()
    processed_lines = []
    for line in lines:
        if "//" in line:
            line = line.split("//")[0].strip()
        if line.strip().startswith("#include"):
            continue
        if "@final" in line:
            line = line.replace("@final", "").strip()
        line = line.replace("unsigned long long", "uint64")
        line = line.replace("unsigned long", "uint32")
        line = line.replace("unsigned short", "uint16")
        line = line.replace("long", "int32")
        line = line.replace("short", "int16")
        line = line.replace("octet", "uint8")
        processed_lines.append(line)
    return "\n".join([line for line in processed_lines if line.strip()])

idl_files = []
for root, _, files in os.walk(idl_dir):
    for file in files:
        if file.endswith(".idl"):
            idl_files.append(os.path.join(root, file))

for idl_path in idl_files:
    msg_name = os.path.splitext(os.path.basename(idl_path))[0]
    try:
        with open(idl_path, "r", encoding="utf-8") as f:
            idl_content = preprocess_idl(f.read())
        typestore.register(get_types_from_idl(idl_content))
    except Exception as e:
        pass
data_by_topic = {topic: [] for topic in topic_configs}

for mcap_file in mcap_files:
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=list(topic_configs.keys())):
            topic = channel.topic
            if topic not in topic_configs:
                continue
            try:
                decoded_msg = typestore.deserialize_cdr(message.data, schema.name)
                config = topic_configs[topic]
                log_entry = {f"{topic}.{field}": func(decoded_msg) for field, func in config["fields"].items()}
                data_by_topic[topic].append(log_entry)
            except Exception as e:
                continue

for topic, config in topic_configs.items():
    csv_file = config["csv_file"]
    data = data_by_topic[topic]
    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file} ({len(data)} rows)")
    else:
        print(f"No data for topic {topic}")