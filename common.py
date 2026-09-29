"""Shared helpers used by every script in this project."""
from pathlib import Path
import numpy as np

DATA_DIR = Path("data/raw")      # where the DroneRF .csv files live
FIG_DIR = Path("figures")        # where plots are saved
FS = 40e6                        # sampling rate in Hz (each DroneRF receiver covers 40 MHz)
CHUNK = 10_000                   # samples per short window (0.25 ms at 40 MHz)


def load_signal(path):
    """Read one DroneRF csv file and return it as a 1-D numpy array."""
    text = Path(path).read_text()
    parts = text.replace("\n", ",").split(",")
    parts = [p for p in parts if p.strip()]          # drop empty pieces
    return np.array(parts, dtype=np.float32)


def list_segments():
    """Find every segment that has both an L file and an H file.

    DroneRF names files like 00000L_3.csv:
      00000 = BUI code (first digit 0 = no drone, 1 = drone)
      L / H = lower or upper half of the 2.4 GHz band
      3     = segment number
    """
    segments = []
    for low_file in sorted(DATA_DIR.glob("*L_*.csv")):
        bui, seg = low_file.stem.split("L_")
        high_file = DATA_DIR / f"{bui}H_{seg}.csv"
        if high_file.exists():
            label = 0 if bui[0] == "0" else 1
            segments.append((low_file, high_file, label, f"{bui}_{seg}"))
    return segments
