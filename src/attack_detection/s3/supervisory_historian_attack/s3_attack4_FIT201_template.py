import os
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIT201_COLUMN = "FIT 201"


def load_fit201_template(
    csv_path: str,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Load the FIT201 template slice ``[start_index, end_index)`` from CSV."""
    df = pd.read_csv(csv_path)
    if FIT201_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{FIT201_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[FIT201_COLUMN].to_numpy(dtype=float)
    if not (0 <= start_index < end_index <= len(values)):
        raise IndexError(
            f"Template range [{start_index}, {end_index}) is out of bounds for "
            f"column length {len(values)}."
        )
    return values[start_index:end_index].copy()


def load_fit201_continuation(
    csv_path: str,
    continuation_start_index: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Load FIT201 values after ``continuation_start_index`` from the same CSV."""
    df = pd.read_csv(csv_path)
    if FIT201_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{FIT201_COLUMN}' not found in CSV; available columns: {list(df.columns)}"
        )

    values = df[FIT201_COLUMN].to_numpy(dtype=float)
    if continuation_start_index >= len(values):
        return np.array([], dtype=float)

    end = len(values) if max_length is None else min(
        len(values), continuation_start_index + max_length
    )
    return values[continuation_start_index:end].copy()


def build_fit201_attack_tail_from_arrays(
    template: np.ndarray,
    continuation: Optional[np.ndarray],
    injection_length: int,
    *,
    start_value: Optional[float] = None,
    attack_progress_span: int = 0,
) -> np.ndarray:
    """
    Build one full FIT201 attack tail of length ``injection_length``.

    Semantics match the final historian-attack requirement:
    - copy the FIT201 attack template first
    - then copy subsequent FIT201 values from the same source CSV
    - if those values are exhausted first, hold the final available value
    - if ``start_value`` is given, the injected sequence starts from that clean
      value and gradually blends into the attack trajectory over the first
      ``attack_progress_span`` samples
    """
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

    if pos < total:
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

    if start_value is not None:
        blend_len = min(total, max(0, int(attack_progress_span)))
        if blend_len == 1:
            result[0] = float(start_value)
        elif blend_len > 1:
            # Always blend against the already-built attack head, even when the
            # template alone fully covers the requested injection tail.
            attack_head = result[:blend_len].copy()
            alphas = np.linspace(0.0, 1.0, blend_len, dtype=float)
            result[:blend_len] = (
                (1.0 - alphas) * float(start_value) + alphas * attack_head
            )

    return result


def build_fit201_attack_tail(
    csv_path: str,
    start_index: int,
    end_index: int,
    injection_length: int,
    *,
    start_value: Optional[float] = None,
    attack_progress_span: int = 0,
) -> np.ndarray:
    """Build one full-tail FIT201 attack directly from template CSV data."""
    template = load_fit201_template(csv_path, start_index, end_index)
    continuation = load_fit201_continuation(
        csv_path,
        end_index,
        max_length=max(0, int(injection_length) - len(template)),
    )
    return build_fit201_attack_tail_from_arrays(
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
    """Plot the original template and one generated full-tail FIT201 attack."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(template, color="tab:blue", linewidth=1.6, label="FIT201 template")
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

    title = "FIT201 template and full-tail attack"
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
    start_value = 0.0
    attack_progress_span = 18

    template_seq = load_fit201_template(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
    )
    generated_tail = build_fit201_attack_tail(
        csv_path=csv_file,
        start_index=template_start_index,
        end_index=template_end_index,
        injection_length=injection_length,
        start_value=start_value,
        attack_progress_span=attack_progress_span,
    )

    save_path = os.path.join(script_dir, "attack_FIT201_template_full_tail.png")
    out = plot_template_and_tail(
        template=template_seq,
        generated_tail=generated_tail,
        save_path=save_path,
        title_suffix=(
            f"(length={injection_length}, start_value={start_value}, "
            f"attack_progress_span={attack_progress_span})"
        ),
    )
    print(f"Saved FIT201 full-tail demo to: {out}")
