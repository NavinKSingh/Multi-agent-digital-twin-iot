"""
Builds data/energy_data.csv from the London Smart Meters dataset
(UK Power Networks, Low Carbon London project, CC BY 4.0):
https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households

Download ONE file of "low-carbon-london-data-168-files" (any of the ~1M-row CSVs),
then run:
    python prepare_dataset.py path/to/block_file.csv [--households 5] [--readings 1000]

Picks the first N households that have `readings` consecutive half-hourly readings
with no gaps and no missing values, and writes the schema used by the environment:
    timestamp, household_id, energy_consumption_kWh
"""
import argparse
import os
import re
import sys

import pandas as pd

try:                                   # works as a script and inside Colab/Jupyter
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()
OUT = os.path.join(BASE, "data", "energy_data.csv")


def find_col(cols, patterns):
    for c in cols:
        if any(re.search(p, c, re.I) for p in patterns):
            return c
    sys.exit(f"Could not find a column matching {patterns} in {list(cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--households", type=int, default=5)
    ap.add_argument("--readings", type=int, default=1000)
    args, _ = ap.parse_known_args()   # ignore Jupyter's own arguments

    files = [p for p in args.csv if os.path.isfile(p) and not p.endswith("kernel.json")]
    if not files:
        sys.exit("No input CSV found. Pass the path of a London Smart Meters block file.")
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    c_id = find_col(df.columns, [r"^LCLid$", r"household"])
    c_t = find_col(df.columns, [r"^DateTime$", r"^tstp$", r"time"])
    c_e = find_col(df.columns, [r"kwh", r"energy"])
    df = df[[c_id, c_t, c_e]].rename(columns={c_id: "hh", c_t: "timestamp", c_e: "kwh"})
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")        # "Null" -> NaN
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna().sort_values(["hh", "timestamp"])
    step = pd.Timedelta(minutes=30)

    chosen = []
    for hh, g in df.groupby("hh"):
        g = g.drop_duplicates("timestamp").reset_index(drop=True)
        # run lengths of consecutive 30-minute readings
        brk = (g["timestamp"].diff() != step).cumsum()
        for _, run in g.groupby(brk):
            if len(run) >= args.readings:
                chosen.append((hh, run.iloc[: args.readings]))
                break
        if len(chosen) == args.households:
            break
    if len(chosen) < args.households:
        sys.exit(f"Only {len(chosen)} households have {args.readings} gap-free readings in this file; "
                 f"try another file or lower --readings.")

    out = []
    for i, (hh, run) in enumerate(chosen, start=1):
        out.append(pd.DataFrame({
            "timestamp": run["timestamp"].values,
            "household_id": i,
            "energy_consumption_kWh": run["kwh"].values,
        }))
        print(f"household {i}: source id {hh}, {run['timestamp'].min()} -> {run['timestamp'].max()}, "
              f"mean {run['kwh'].mean():.3f} kWh/30min")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.concat(out).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
