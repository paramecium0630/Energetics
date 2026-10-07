"""Read canonical node energetics, with support for archived legacy CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd


def read_dynamics_parameters(directory, node_layers):
    """Read per-node dynamics; never infer archived parameters from current input."""
    path = Path(directory) / "dynamics_parameters.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate this dataset with dynamics metadata "
            "or restore the parameters recorded for that run; current .nml is not a valid substitute."
        )
    data = pd.read_csv(path)
    required = ["Node", "Layer", "r", "noise"]
    if not set(required) <= set(data.columns):
        raise ValueError(f"Missing dynamics columns in {path}")
    data = data[required].sort_values("Node").reset_index(drop=True)
    if data.empty or data["Node"].duplicated().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"Invalid dynamics rows in {path}")
    mapping = node_layers[["Node", "Layer"]].sort_values("Node").reset_index(drop=True)
    if not data[["Node", "Layer"]].equals(mapping):
        raise ValueError(f"Dynamics and node-layer mapping differ in {path}")
    if (data["noise"] <= 0).any():
        raise ValueError(f"Nonpositive noise in {path}")
    return data


def require_matching_dynamics(directory, reference_dir, node_layers):
    actual = read_dynamics_parameters(directory, node_layers)
    reference = read_dynamics_parameters(reference_dir, node_layers)
    for column in ("r", "noise"):
        if not np.allclose(actual[column], reference[column], rtol=1e-12, atol=0.0):
            raise ValueError(f"Dynamics mismatch ({column}): {directory} versus {reference_dir}")


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
