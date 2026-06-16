import argparse
import glob
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
NUMERIC_COLUMNS = [
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
    "normalized_car_position",
    "wind_speed",
    "wind_direction",
]


def resolve_paths(patterns, new_csv):
    paths = []
    for pattern in patterns:
        if not Path(pattern).is_absolute():
            paths.extend(ROOT_DIR.glob(pattern))
        else:
            paths.extend(Path(match) for match in glob.glob(pattern))

    resolved_new = new_csv.resolve()
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved == resolved_new or resolved in seen or "new_recordings" in resolved.parts:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


def load_reference(paths, sample_rows):
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skip {path}: {exc}")
            continue
        missing = [col for col in NUMERIC_COLUMNS if col not in df.columns]
        if missing:
            print(f"Skip {path}: colonne mancanti {missing}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("Nessun CSV di riferimento valido trovato.")

    reference = pd.concat(frames, ignore_index=True)
    if sample_rows and len(reference) > sample_rows:
        reference = reference.sample(sample_rows, random_state=42)
    return reference


def describe_numeric(df, columns):
    summary = df[columns].describe(percentiles=[0.05, 0.5, 0.95]).T
    return summary.rename(columns={"50%": "median"})


def compare(reference, current):
    columns = [col for col in NUMERIC_COLUMNS if col in reference.columns and col in current.columns]
    ref = describe_numeric(reference, columns)
    cur = describe_numeric(current, columns)

    rows = []
    for col in columns:
        ref_std = ref.loc[col, "std"]
        median_delta = cur.loc[col, "median"] - ref.loc[col, "median"]
        z_delta = median_delta / ref_std if pd.notna(ref_std) and ref_std > 1e-9 else 0.0
        out_low = (current[col] < ref.loc[col, "min"]).mean() * 100.0
        out_high = (current[col] > ref.loc[col, "max"]).mean() * 100.0
        rows.append(
            {
                "column": col,
                "new_median": cur.loc[col, "median"],
                "ref_median": ref.loc[col, "median"],
                "median_delta": median_delta,
                "delta_in_ref_std": z_delta,
                "new_p05": cur.loc[col, "5%"],
                "ref_p05": ref.loc[col, "5%"],
                "new_p95": cur.loc[col, "95%"],
                "ref_p95": ref.loc[col, "95%"],
                "new_std": cur.loc[col, "std"],
                "ref_std": ref_std,
                "pct_below_ref_min": out_low,
                "pct_above_ref_max": out_high,
            }
        )

    result = pd.DataFrame(rows)
    result["suspicion_score"] = (
        result["delta_in_ref_std"].abs()
        + result["pct_below_ref_min"] / 10.0
        + result["pct_above_ref_max"] / 10.0
    )
    return result.sort_values("suspicion_score", ascending=False)


def print_label_distribution(name, df):
    if "result" not in df.columns:
        return
    print(f"\nDistribuzione label {name}:")
    print(df["result"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")


def parse_args():
    parser = argparse.ArgumentParser(description="Confronta una registrazione nuova con i CSV storici.")
    parser.add_argument("new_csv", type=Path, help="CSV registrato con record_shared_memory.py")
    parser.add_argument(
        "--reference",
        nargs="+",
        default=["data/*.csv", "data/*/*.csv"],
        help="Pattern dei CSV storici.",
    )
    parser.add_argument("--sample-reference", type=int, default=250000, help="Campiona al massimo N righe storiche.")
    parser.add_argument("--top", type=int, default=15, help="Numero di colonne sospette da stampare.")
    return parser.parse_args()


def main():
    args = parse_args()
    new_csv = args.new_csv if args.new_csv.is_absolute() else ROOT_DIR / args.new_csv
    current = pd.read_csv(new_csv)
    reference_paths = resolve_paths(args.reference, new_csv)
    reference = load_reference(reference_paths, args.sample_reference)

    print(f"Nuovo CSV: {new_csv}")
    print(f"Righe nuovo CSV: {len(current)}")
    print(f"CSV riferimento: {len(reference_paths)} file, {len(reference)} righe usate")

    print_label_distribution("nuova", current)
    print_label_distribution("storica", reference)

    comparison = compare(reference, current)
    columns_to_show = [
        "column",
        "new_median",
        "ref_median",
        "delta_in_ref_std",
        "new_p05",
        "ref_p05",
        "new_p95",
        "ref_p95",
        "pct_below_ref_min",
        "pct_above_ref_max",
    ]
    print("\nColonne più sospette:")
    print(comparison[columns_to_show].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
