import os
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIT301_TEMPLATE_COLUMN = "FIT 301"
FIT301_LEARNED_RISE_STAGE = np.array(
    [
        0.000512443,
        0.002946546,
        0.022803702,
        0.044582516,
        0.06610511,
        0.08826826,
        0.117221273,
        0.179739282,
        0.325529248,
        0.53947407,
        0.777375638,
        1.057938,
        1.31198144,
        1.5208019,
        1.64955318,
        1.72193563,
    ],
    dtype=float,
)
FIT301_RISE_BRIDGE_STEPS = 4
FIT301_HIGH_ENTRY_BRIDGE_STEPS = 12


def load_fit301_template(
    csv_path: str,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Load the FIT301 template slice ``[start_index, end_index)`` from CSV."""
    df = pd.read_csv(csv_path)
    if FIT301_TEMPLATE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{FIT301_TEMPLATE_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[FIT301_TEMPLATE_COLUMN].to_numpy(dtype=float)
    if not (0 <= start_index < end_index <= len(values)):
        raise IndexError(
            f"Template range [{start_index}, {end_index}) is out of bounds for "
            f"column length {len(values)}."
        )
    return values[start_index:end_index].copy()


def load_fit301_continuation(
    csv_path: str,
    continuation_start_index: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Load FIT301 values after ``continuation_start_index`` from the same CSV."""
    df = pd.read_csv(csv_path)
    if FIT301_TEMPLATE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{FIT301_TEMPLATE_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[FIT301_TEMPLATE_COLUMN].to_numpy(dtype=float)
    if continuation_start_index >= len(values):
        return np.array([], dtype=float)

    end = len(values) if max_length is None else min(
        len(values), continuation_start_index + max_length
    )
    return values[continuation_start_index:end].copy()


def _linear_bridge(start_value: float, end_value: float, steps: int) -> np.ndarray:
    if steps <= 0 or np.isclose(start_value, end_value):
        return np.array([], dtype=float)
    return np.linspace(
        float(start_value),
        float(end_value),
        int(steps) + 2,
        dtype=float,
    )[1:-1]


def _build_fit301_response_trace(
    template: np.ndarray,
    continuation: Optional[np.ndarray],
    *,
    learned_rise_stage: Optional[np.ndarray],
    entry_value: Optional[float],
) -> np.ndarray:
    template = np.asarray(template, dtype=float)
    if template.ndim != 1 or len(template) == 0:
        raise ValueError("template must be a non-empty 1D sequence.")

    continuation_arr = (
        np.asarray(continuation, dtype=float)
        if continuation is not None
        else np.array([], dtype=float)
    )
    if continuation_arr.ndim != 1:
        raise ValueError("continuation must be a 1D sequence when provided.")

    source_tail = np.concatenate([template, continuation_arr]) if len(continuation_arr) > 0 else template
    segments: list[np.ndarray] = []

    if entry_value is None:
        if learned_rise_stage is not None and len(learned_rise_stage) > 0:
            segments.append(np.asarray(learned_rise_stage, dtype=float))
        segments.append(source_tail)
        return np.concatenate(segments)

    entry = float(entry_value)
    rise_stage = (
        np.array([], dtype=float)
        if learned_rise_stage is None
        else np.asarray(learned_rise_stage, dtype=float)
    )
    if rise_stage.ndim != 1:
        raise ValueError("learned_rise_stage must be a 1D sequence when provided.")

    if len(rise_stage) > 0 and entry <= float(rise_stage[-1]):
        rise_start = int(np.argmin(np.abs(rise_stage - entry)))
        rise_suffix = rise_stage[rise_start:]
        if len(rise_suffix) > 0:
            bridge = _linear_bridge(entry, float(rise_suffix[0]), FIT301_RISE_BRIDGE_STEPS)
            if len(bridge) > 0:
                segments.append(bridge)
            segments.append(rise_suffix)
        segments.append(source_tail)
        return np.concatenate(segments)

    bridge = _linear_bridge(entry, float(source_tail[0]), FIT301_HIGH_ENTRY_BRIDGE_STEPS)
    if len(bridge) > 0:
        segments.append(bridge)
    segments.append(source_tail)
    return np.concatenate(segments)


def build_fit301_attack_tail_from_arrays(
    template: np.ndarray,
    continuation: Optional[np.ndarray],
    injection_length: int,
    *,
    learned_rise_stage: Optional[np.ndarray] = None,
    entry_value: Optional[float] = None,
) -> np.ndarray:
    """
    Build one full-tail FIT301 directly-affected-PV attack from template-source values.

    When ``learned_rise_stage`` is provided, prepend that learned onset segment
    before replaying the template slice itself. By default no synthetic bridge
    is inserted.
    """
    if injection_length <= 0:
        raise ValueError("injection_length must be positive.")

    response_trace = _build_fit301_response_trace(
        template,
        continuation,
        learned_rise_stage=learned_rise_stage,
        entry_value=entry_value,
    )
    result = np.empty(int(injection_length), dtype=float)
    take_count = min(len(result), len(response_trace))
    result[:take_count] = response_trace[:take_count]
    if take_count < len(result):
        result[take_count:] = response_trace[-1]
    return result


def build_fit301_attack_tail(
    csv_path: str,
    start_index: int,
    end_index: int,
    injection_length: int,
    *,
    learned_rise_stage: Optional[np.ndarray] = None,
    entry_value: Optional[float] = None,
) -> np.ndarray:
    """Build one full-tail FIT301 attack directly from template CSV data."""
    template = load_fit301_template(csv_path, start_index, end_index)
    continuation = load_fit301_continuation(
        csv_path,
        end_index,
        max_length=None,
    )
    return build_fit301_attack_tail_from_arrays(
        template,
        continuation,
        injection_length,
        learned_rise_stage=learned_rise_stage,
        entry_value=entry_value,
    )


def plot_fit301_tail(
    template: np.ndarray,
    generated_tail: np.ndarray,
    save_path: str,
    title_suffix: str = "",
) -> str:
    """Plot the original FIT301 template and one generated attack tail."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(template, color="tab:blue", linewidth=1.6, label="FIT301 template")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        generated_tail,
        color="tab:green",
        linewidth=1.6,
        label="Generated FIT301 attack",
    )
    axes[1].set_xlabel("Local index from injection start")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    title = "FIT301 template and attack tail"
    if title_suffix:
        title = f"{title} {title_suffix}"
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    csv_file = os.path.join(data_dir, "SWaT_dataset_Jul 19 v2_dealed.csv")
    template_start_index = 11235
    template_end_index = 13936
    injection_length = 1860

    template_seq = load_fit301_template(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
    )
    generated_tail = build_fit301_attack_tail(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
        injection_length=injection_length,
    )
    save_path = os.path.join(script_dir, "attack_FIT301_template_full_tail.png")
    out = plot_fit301_tail(
        template=template_seq,
        generated_tail=generated_tail,
        save_path=save_path,
        title_suffix=f"(length={injection_length})",
    )
    print(f"Saved FIT301 full-tail demo to: {out}")
