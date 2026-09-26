"""Read canonical node energetics, with support for archived legacy CSVs."""
from pathlib import Path
import pandas as pd


def read_node_energetics(path):
    path = Path(path)
    with path.open() as stream:
        first_line = stream.readline().strip()
    # New output starts with the CSV header; legacy output starts with a title.
    skiprows = 0 if first_line.startswith("Node,Layer,") else 1
    data = pd.read_csv(path, skiprows=skiprows)
    data.columns = data.columns.str.strip()
    return data.rename(columns={
        "Node": "node", "Layer": "layer", "HR": "heat_rate",
        "EPR": "entropy_rate", "WR": "work_rate", "UR": "internal_rate",
    })
