import os
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl
import pandas as pd

mcap_files = [
    "../logall/rosbag-20250403T141242-08-control/rosbag-20250403T141242-08-control_0.mcap",
    "../logall/rosbag-20250403T141242-08-control/rosbag-20250403T141242-08-control_1.mcap"
]

perception_ego_state_idl = """
module safead_ros2_msgs { 
    module msg {
        @final struct EgoState {
            unsigned long long timestamp_nanoseconds;
            float velocity_x;
            float wheels_toe_angle_fl;
            float wheels_toe_angle_fr;
            float angular_rate_z;
            float acceleration_x;
            float acceleration_y;
        };
    };
};
"""

topic = "/observer/perception_ego_state"
csv_file = "perception_ego_state_data.csv"

def preprocess_idl(content):
    lines = content.splitlines()
    processed = []
    for line in lines:
        if "//" in line:
            line = line.split("//")[0].strip()
        if "@final" in line:
            line = line.replace("@final", "").strip()
        if line.strip().startswith("#include"):
            continue
        line = line.replace("unsigned long long", "uint64")
        line = line.replace("unsigned long", "uint32")
        line = line.replace("unsigned short", "uint16")
        line = line.replace("long", "int32")
        line = line.replace("short", "int16")
        line = line.replace("octet", "uint8")
        processed.append(line)
    return "\n".join([l for l in processed if l.strip()])

typestore = get_typestore(Stores.ROS2_HUMBLE)
idl_clean = preprocess_idl(perception_ego_state_idl)
typestore.register(get_types_from_idl(idl_clean))

extract_fields = {
    "timestamp_nanoseconds": lambda msg: msg.timestamp_nanoseconds,
    "velocity_x": lambda msg: msg.velocity_x,
    "wheels_toe_angle_fl": lambda msg: msg.wheels_toe_angle_fl,
    "wheels_toe_angle_fr": lambda msg: msg.wheels_toe_angle_fr,
    "angular_rate_z": lambda msg: msg.angular_rate_z,
    "acceleration_x": lambda msg: msg.acceleration_x,
    "acceleration_y": lambda msg: msg.acceleration_y,
}

data = []

for mcap_file in mcap_files:
    print(f"Elaboration {mcap_file}")
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            try:
                decoded = typestore.deserialize_cdr(message.data, schema.name)
                prefix = topic
                row = {f"{prefix}.{k}": func(decoded) for k, func in extract_fields.items()}
                data.append(row)
            except Exception as e:
                continue

if data:
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Saved {csv_file} with {len(df)} rows")
else:
    print("No data extracted for topic:", topic)
