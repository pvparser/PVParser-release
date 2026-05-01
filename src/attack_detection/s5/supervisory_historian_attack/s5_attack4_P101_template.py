import os
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P101_TEMPLATE_COLUMN = "P101 Status"


def load_p101_template(
    csv_path: str,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Load the P101 template slice ``[start_index, end_index)`` from CSV."""
    df = pd.read_csv(csv_path)
    if P101_TEMPLATE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{P101_TEMPLATE_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[P101_TEMPLATE_COLUMN].to_numpy(dtype=float)
    if not (0 <= start_index < end_index <= len(values)):
        raise IndexError(
            f"Template range [{start_index}, {end_index}) is out of bounds for "
            f"column length {len(values)}."
        )
    return values[start_index:end_index].copy()


def load_p101_continuation(
    csv_path: str,
    continuation_start_index: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Load P101 values after ``continuation_start_index`` from the same CSV."""
    df = pd.read_csv(csv_path)
    if P101_TEMPLATE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{P101_TEMPLATE_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[P101_TEMPLATE_COLUMN].to_numpy(dtype=float)
    if continuation_start_index >= len(values):
        return np.array([], dtype=float)

    end = len(values) if max_length is None else min(
        len(values), continuation_start_index + max_length
    )
    return values[continuation_start_index:end].copy()


def build_p101_attack_tail_from_arrays(
    template: np.ndarray,
    continuation: Optional[np.ndarray],
    injection_length: int,
) -> np.ndarray:
    """Build one full-tail P101 manipulated-PV attack from template-source values."""
    template = np.asarray(template, dtype=float)
    if template.ndim != 1 or len(template) == 0:
        raise ValueError("template must be a non-empty 1D sequence.")
    if injection_length <= 0:
        raise ValueError("injection_length must be positive.")

    result = np.empty(int(injection_length), dtype=float)
    total = len(result)

    take_template = min(total, len(template))
    result[:take_template] = template[:take_template]
    pos = take_template
    if pos >= total:
        return result

    continuation_arr = (
        np.asarray(continuation, dtype=float)
        if continuation is not None
        else np.array([], dtype=float)
    )
    if len(continuation_arr) > 0:
        take_cont = min(total - pos, len(continuation_arr))
        result[pos : pos + take_cont] = continuation_arr[:take_cont]
        pos += take_cont
        if pos < total:
            result[pos:] = continuation_arr[-1]
    else:
        result[pos:] = template[-1]

    return result


def build_p101_attack_tail(
    csv_path: str,
    start_index: int,
    end_index: int,
    injection_length: int,
) -> np.ndarray:
    """Build one full-tail P101 attack directly from template CSV data."""
    template = load_p101_template(csv_path, start_index, end_index)
    continuation = load_p101_continuation(
        csv_path,
        end_index,
        max_length=max(0, int(injection_length) - len(template)),
    )
    return build_p101_attack_tail_from_arrays(template, continuation, injection_length)


def plot_p101_tail(
    template: np.ndarray,
    generated_tail: np.ndarray,
    save_path: str,
    title_suffix: str = "",
) -> str:
    """Plot the original P101 template and one generated attack tail."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(template, color="tab:purple", linewidth=1.6, label="P101 template")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        generated_tail,
        color="tab:brown",
        linewidth=1.6,
        label="Generated P101 attack",
    )
    axes[1].set_xlabel("Local index from injection start")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    title = "P101 template and attack tail"
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

    template_seq = load_p101_template(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
    )
    generated_tail = build_p101_attack_tail(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
        injection_length=injection_length,
    )
    save_path = os.path.join(script_dir, "attack_P101_template_full_tail.png")
    out = plot_p101_tail(
        template=template_seq,
        generated_tail=generated_tail,
        save_path=save_path,
        title_suffix=f"(length={injection_length})",
    )
    print(f"Saved P101 full-tail demo to: {out}")
