import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fiber-analysis-matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _histogram(values, path: Path, title: str, xlabel: str, bins, value_range=None):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"{title}\n{len(values)} accepted fibers", fontsize=15, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of fibers")
    ax.grid(axis="y", alpha=.25)
    if values:
        counts, edges, bars = ax.hist(
            values,
            bins=bins,
            range=value_range,
            color="#2878B5",
            edgecolor="white",
            linewidth=.8,
        )
        ax.bar_label(bars, labels=[str(int(count)) if count else "" for count in counts], padding=3, fontsize=9)
        ax.set_ylim(0, max(counts) * 1.15 + .5)
    else:
        ax.text(.5, .5, "No accepted fibers detected", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    if value_range:
        ax.set_xlim(*value_range)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_result(output: Path, stem: str, method: str, fibers, debug: dict, keep_debug: bool, unit_name: str) -> dict:
    folder = output / stem / method
    folder.mkdir(parents=True, exist_ok=True)
    accepted = [fiber for fiber in fibers if not fiber.rejected]
    calibrated = any(fiber.calibrated_length is not None for fiber in accepted)
    lengths = [fiber.calibrated_length if calibrated else fiber.centerline_length_px for fiber in accepted]
    length_unit = unit_name if calibrated else "source-image pixels"
    _histogram(
        lengths,
        folder / "length_distribution.png",
        f"{stem} — {method}\nFiber centerline-length distribution",
        f"Geodesic length ({length_unit})",
        bins=20,
    )
    _histogram(
        [fiber.orientation_deg for fiber in accepted],
        folder / "orientation_distribution.png",
        f"{stem} — {method}\nAxial fiber-orientation distribution",
        "Orientation (degrees; 0° horizontal, 90° vertical)",
        bins=np.arange(0, 181, 10),
        value_range=(0, 180),
    )
    if keep_debug:
        for name, array in debug.items():
            np.save(folder / f"{name}.npy", array)
    return {"accepted": len(accepted), "rejected": len(fibers) - len(accepted)}
