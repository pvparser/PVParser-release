import os
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIT301_TEMPLATE_COLUMN = "LIT 301"
LIT301_REFERENCE_PRE_ATTACK_VALUE = 878.5408
LIT301_REFERENCE_ATTACKED_VALUE = 789.300842
LIT301_ATTACK_END_OFFSET = 261


def load_lit301_template(
    csv_path: str,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Load the LIT301 template slice ``[start_index, end_index)`` from CSV."""
    df = pd.read_csv(csv_path)
    if LIT301_TEMPLATE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{LIT301_TEMPLATE_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[LIT301_TEMPLATE_COLUMN].to_numpy(dtype=float)
    if not (0 <= start_index < end_index <= len(values)):
        raise IndexError(
            f"Template range [{start_index}, {end_index}) is out of bounds for "
            f"column length {len(values)}."
        )
    return values[start_index:end_index].copy()


def load_lit301_continuation(
    csv_path: str,
    continuation_start_index: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Load LIT301 values after ``continuation_start_index`` from the same CSV."""
    df = pd.read_csv(csv_path)
    if LIT301_TEMPLATE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{LIT301_TEMPLATE_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[LIT301_TEMPLATE_COLUMN].to_numpy(dtype=float)
    if continuation_start_index >= len(values):
        return np.array([], dtype=float)

    end = len(values) if max_length is None else min(
        len(values), continuation_start_index + max_length
    )
    return values[continuation_start_index:end].copy()


def build_lit301_attack_tail_from_arrays(
    template: np.ndarray,
    continuation: Optional[np.ndarray],
    injection_length: int,
    *,
    pre_attack_value: Optional[float] = None,
    attack_end_offset: int = LIT301_ATTACK_END_OFFSET,
    reference_pre_attack_value: float = LIT301_REFERENCE_PRE_ATTACK_VALUE,
    reference_attacked_value: float = LIT301_REFERENCE_ATTACKED_VALUE,
) -> np.ndarray:
    """
    Build one full-tail LIT301 manipulated-PV attack from template-source values.

    Attack2 LIT301 has two phases:
    - the active spoof segment uses the template attack value directly
    - once the spoof ends, the post-attack recovery segment is shifted by the
      difference between the target clean pre-attack value and the reference
      pre-attack value, so the post-attack drop amplitude is preserved
    """
    template = np.asarray(template, dtype=float)
    if template.ndim != 1 or len(template) == 0:
        raise ValueError("template must be a non-empty 1D sequence.")
    if injection_length <= 0:
        raise ValueError("injection_length must be positive.")

    continuation_arr = (
        np.asarray(continuation, dtype=float)
        if continuation is not None
        else np.array([], dtype=float)
    )

    source = (
        np.concatenate([template, continuation_arr])
        if len(continuation_arr) > 0
        else template.copy()
    )
    if pre_attack_value is not None:
        split = max(0, min(int(attack_end_offset), len(source)))
        shift = float(pre_attack_value) - float(reference_pre_attack_value)
        if split < len(source):
            source = source.copy()
            source[split:] = source[split:] + shift

    result = np.empty(int(injection_length), dtype=float)
    total = len(result)
    take_source = min(total, len(source))
    result[:take_source] = source[:take_source]
    if take_source < total:
        result[take_source:] = source[-1]

    return result


def build_lit301_attack_tail(
    csv_path: str,
    start_index: int,
    end_index: int,
    injection_length: int,
    *,
    pre_attack_value: Optional[float] = None,
    attack_end_offset: int = LIT301_ATTACK_END_OFFSET,
    reference_pre_attack_value: float = LIT301_REFERENCE_PRE_ATTACK_VALUE,
    reference_attacked_value: float = LIT301_REFERENCE_ATTACKED_VALUE,
) -> np.ndarray:
    """Build one full-tail LIT301 attack directly from template CSV data."""
    template = load_lit301_template(csv_path, start_index, end_index)
    continuation = load_lit301_continuation(
        csv_path,
        end_index,
        max_length=max(0, int(injection_length) - len(template)),
    )
    return build_lit301_attack_tail_from_arrays(
        template,
        continuation,
        injection_length,
        pre_attack_value=pre_attack_value,
        attack_end_offset=attack_end_offset,
        reference_pre_attack_value=reference_pre_attack_value,
        reference_attacked_value=reference_attacked_value,
    )


def plot_lit301_tail(
    template: np.ndarray,
    generated_tail: np.ndarray,
    save_path: str,
    title_suffix: str = "",
) -> str:
    """Plot the original LIT301 template and one generated attack tail."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(template, color="tab:orange", linewidth=1.6, label="LIT301 template")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        generated_tail,
        color="tab:red",
        linewidth=1.6,
        label="Generated LIT301 attack",
    )
    axes[1].set_xlabel("Local index from injection start")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    title = "LIT301 template and attack tail"
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

    template_seq = load_lit301_template(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
    )
    generated_tail = build_lit301_attack_tail(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
        injection_length=injection_length,
    )
    save_path = os.path.join(script_dir, "attack_LIT301_template_full_tail.png")
    out = plot_lit301_tail(
        template=template_seq,
        generated_tail=generated_tail,
        save_path=save_path,
        title_suffix=f"(length={injection_length})",
    )
    print(f"Saved LIT301 full-tail demo to: {out}")
