import os
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIT301_COLUMN = "LIT 301"


def load_lit301_template(
    csv_path: str,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Load the LIT301 template slice ``[start_index, end_index)`` from CSV."""
    df = pd.read_csv(csv_path)
    if LIT301_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{LIT301_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[LIT301_COLUMN].to_numpy(dtype=float)
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
    if LIT301_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{LIT301_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[LIT301_COLUMN].to_numpy(dtype=float)
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
    start_value: Optional[float] = None,
    attack_progress_span: int = 0,
) -> np.ndarray:
    """
    Build one full LIT301 attack tail of length ``injection_length``.

    Semantics:
    - if ``start_value`` and ``attack_progress_span`` are provided, use that
      span as an attack-progress ramp learned from the template-source rising
      trend itself
    - after the progress ramp, continue replaying the remaining template-source
      values
    - if the template-source values are exhausted, hold the final available
      value
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

    result = np.empty(int(injection_length), dtype=float)
    total = len(result)

    progress_len = 0
    if start_value is not None:
        progress_len = min(total, max(0, int(attack_progress_span)), len(source))
        if progress_len == 1:
            result[0] = float(start_value)
        elif progress_len > 1:
            progress_source = source[:progress_len]
            src_start = float(progress_source[0])
            src_end = float(progress_source[-1])
            src_delta = src_end - src_start
            target_end = src_end
            progress = np.empty(progress_len, dtype=float)
            progress[0] = float(start_value)
            if abs(src_delta) <= 1e-12:
                progress[1:] = target_end
            else:
                scale = (target_end - float(start_value)) / src_delta
                for idx in range(1, progress_len):
                    progress[idx] = (
                        progress[idx - 1]
                        + scale * float(progress_source[idx] - progress_source[idx - 1])
                    )
            result[:progress_len] = progress

    pos = progress_len
    remaining_source = source[progress_len:]
    if pos >= total:
        return result

    take_source = min(total - pos, len(remaining_source))
    result[pos : pos + take_source] = remaining_source[:take_source]
    pos += take_source
    if pos >= total:
        return result

    result[pos:] = source[-1]

    return result


def build_lit301_attack_tail(
    csv_path: str,
    start_index: int,
    end_index: int,
    injection_length: int,
    *,
    start_value: Optional[float] = None,
    attack_progress_span: int = 0,
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
        start_value=start_value,
        attack_progress_span=attack_progress_span,
    )


def plot_template_and_tail(
    template: np.ndarray,
    generated_tail: np.ndarray,
    save_path: str,
    title_suffix: str = "",
) -> str:
    """Plot the original template and one generated full-tail attack sequence."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(template, color="tab:blue", linewidth=1.6, label="LIT301 template")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        generated_tail,
        color="tab:green",
        linewidth=1.6,
        label="Generated full-tail attack",
    )
    axes[1].set_xlabel("Local index from injection start")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    title = "LIT301 template and full-tail attack"
    if title_suffix:
        title = f"{title} {title_suffix}"
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "data", "SWaT_dataset_Jul 19 v2_dealed.csv")

    template_start_index = 11235
    template_end_index = 13936
    injection_length = 1860
    start_value = 1008.0
    attack_progress_span = 448

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
        start_value=start_value,
        attack_progress_span=attack_progress_span,
    )

    save_path = os.path.join(script_dir, "attack_LIT301_template_full_tail.png")
    out = plot_template_and_tail(
        template=template_seq,
        generated_tail=generated_tail,
        save_path=save_path,
        title_suffix=(
            f"(length={injection_length}, start_value={start_value}, "
            f"attack_progress_span={attack_progress_span})"
        ),
    )
    print(f"Saved LIT301 full-tail demo to: {out}")
