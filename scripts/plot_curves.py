#!/usr/bin/env python3
"""Plot training loss and accept metrics from speculators train logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_train_log(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore")
    flat = re.sub(r"\n\s+", " ", raw)
    pattern = re.compile(
        r"train/loss=(?P<loss>[\d.eE+-]+).*?"
        r"train/ce_loss=(?P<ce>[\d.eE+-]+).*?"
        r"train/tv_loss=(?P<tv>[\d.eE+-]+).*?"
        r"train/accept_rate=(?P<ar>[\d.eE+-]+).*?"
        r"train/accept_len=(?P<alen>[\d.eE+-]+).*?"
        r"global_step=(?P<step>\d+)",
        re.S,
    )
    by_step: dict[int, dict[str, float]] = {}
    for match in pattern.finditer(flat):
        step = int(match.group("step"))
        by_step[step] = {
            "loss": float(match.group("loss")),
            "ce_loss": float(match.group("ce")),
            "tv_loss": float(match.group("tv")),
            "accept_rate": float(match.group("ar")),
            "accept_len": float(match.group("alen")),
        }
    if not by_step:
        raise ValueError(f"No training metrics found in {path}")

    steps = np.array(sorted(by_step))
    return {
        "step": steps,
        "loss": np.array([by_step[s]["loss"] for s in steps]),
        "ce_loss": np.array([by_step[s]["ce_loss"] for s in steps]),
        "tv_loss": np.array([by_step[s]["tv_loss"] for s in steps]),
        "accept_rate": np.array([by_step[s]["accept_rate"] for s in steps]),
        "accept_len": np.array([by_step[s]["accept_len"] for s in steps]),
    }


def ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    smoothed: float | None = None
    for i, value in enumerate(values):
        smoothed = value if smoothed is None else alpha * value + (1 - alpha) * smoothed
        out[i] = smoothed
    return out


def _plot_raw_ema(
    ax: plt.Axes,
    steps: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    annotate_final: bool = True,
) -> None:
    ema_values = ema(values)
    ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.8, label="raw")
    ax.plot(steps, ema_values, color=color, linewidth=2.0, label="EMA α=0.1")
    if annotate_final:
        ax.annotate(
            f"{ema_values[-1]:.3f}",
            xy=(steps[-1], ema_values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=color,
            va="center",
        )


def plot_curves(
    data: dict[str, np.ndarray],
    *,
    title: str,
    accept_len_target: float,
    output: Path,
) -> None:
    steps = data["step"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_loss = axes[0]
    _plot_raw_ema(ax_loss, steps, data["loss"], color="#1f77b4")
    ax_loss.plot(steps, ema(data["ce_loss"]), color="#ff7f0e", linewidth=1.8, label="ce EMA α=0.1")
    ax_loss.plot(steps, ema(data["tv_loss"]), color="#2ca02c", linewidth=1.8, label="tv EMA α=0.1")
    ax_loss.set_ylabel("train/loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=9)

    ax_ar = axes[1]
    _plot_raw_ema(ax_ar, steps, data["accept_rate"], color="#1f77b4")
    ax_ar.set_ylabel("train/accept_rate")
    ax_ar.set_ylim(0.0, max(0.75, float(data["accept_rate"].max()) * 1.05))
    ax_ar.grid(True, alpha=0.3)
    ax_ar.legend(loc="lower right", fontsize=9)

    ax_alen = axes[2]
    _plot_raw_ema(ax_alen, steps, data["accept_len"], color="#d62728")
    ax_alen.axhline(
        accept_len_target,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"target={accept_len_target:g}",
    )
    ax_alen.set_ylabel("train/accept_len")
    ax_alen.set_xlabel("global_step")
    ax_alen.set_ylim(
        1.0,
        max(accept_len_target + 0.5, float(data["accept_len"].max()) * 1.05),
    )
    ax_alen.grid(True, alpha=0.3)
    ax_alen.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Training log path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: log/<log_stem>_curves.png)",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    parser.add_argument(
        "--accept-len-target",
        type=float,
        default=5.0,
        help="Horizontal target line for accept_len (default: block_size=8)",
    )
    args = parser.parse_args()

    data = parse_train_log(args.log)
    output = args.output or args.log.with_name(f"{args.log.stem}_curves.png")
    n = len(data["step"])
    title = args.title or (
        f"815 glm5_dspark train metrics (1node 225) n={n}, "
        f"step {int(data['step'][0])} → {int(data['step'][-1])}"
    )
    plot_curves(
        data,
        title=title,
        accept_len_target=args.accept_len_target,
        output=output,
    )
    print(f"Saved {output}")
    print(
        f"steps={n} last_loss={data['loss'][-1]:.3f} "
        f"last_ar={data['accept_rate'][-1]:.3f} "
        f"last_alen={data['accept_len'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
