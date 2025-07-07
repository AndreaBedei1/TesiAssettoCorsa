from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl
import pandas as pd

# File .mcap
mcap_files = [
    "../logall/rosbag-20250403T141242-08-control/rosbag-20250403T141242-08-control_0.mcap",
    "../logall/rosbag-20250403T141242-08-control/rosbag-20250403T141242-08-control_1.mcap"
]

# Topic and output
topic = "/controller/control"
csv_file = "control_brake_data.csv"

idl_car_control = """
module common_msgs {
  module msg {
    struct Timestamp {
      uint64 nanoseconds;
    };

    struct WheelsData {
      float fl;
      float fr;
      float rl;
      float rr;
    };
  };
};

module musa {
  module msg {
    struct CarControl {
      common_msgs::msg::Timestamp timestamp;
      float steering_target_percent;
      uint8 steer_mode;
      common_msgs::msg::WheelsData brake_target_percent;
      boolean brake_prefill_request;
      float throttle_target_percent;
      uint8 gear_target;
    };
  };
};
"""

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
typestore.register(get_types_from_idl(preprocess_idl(idl_car_control)))

extract_fields = {
    "timestamp_nanoseconds": lambda msg: msg.timestamp.nanoseconds,
    "brake_target_percent_fl": lambda msg: msg.brake_target_percent.fl,
    "brake_target_percent_fr": lambda msg: msg.brake_target_percent.fr,
    "brake_target_percent_rl": lambda msg: msg.brake_target_percent.rl,
    "brake_target_percent_rr": lambda msg: msg.brake_target_percent.rr,
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
                print(f"Error at {mcap_file}: {e}")
                continue

if data:
    df = pd.DataFrame(data)
    df = df.sort_values("/controller/control.timestamp_nanoseconds").reset_index(drop=True)
    df.to_csv(csv_file, index=False)
    print(f"Saved {csv_file} with {len(df)} rows")
else:
    print(f"No data for the topic {topic}")
