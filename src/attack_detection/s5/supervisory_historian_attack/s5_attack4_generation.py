"""
Build outputs directly under ``../data/`` for the s5 supervisory-historian script.

Program semantics:
- ``training.csv``: rows ``[train_start, train_end)``, no injection.
- ``test_base.csv``: rows ``[test_start, test_end)``, no injection.
- ``test_XX.csv``: supervisory-view attack files; one shared injection point
  per file.
- ``_control_attack_csv/test_XX.csv``: control-layer attack source files with
  manipulated PVs kept at attack values.
- The shared injection point is sampled only from ``[inject_start, inject_end)``
  inside the test slice.
- In supervisory-view ``test_XX.csv``, manipulated PVs remain at clean
  ``test_base`` values. In ``_control_attack_csv/test_XX.csv``, manipulated PV
  template replacement begins at the injection point.
- Directly affected PVs start from the clean value at the injection point and
  gradually enter the attack trajectory, then keep following the attack script
  to the end of the test slice in both views.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
SHARED_ROOT = Path(__file__).resolve().parents[1]
for import_root in (MODULE_ROOT, SHARED_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _resolve_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        src_root = parent / "src"
        if (src_root / "attack_detection").is_dir() and (src_root / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing src/attack_detection.")

try:
    from .attack_injection_logging import write_attack_injection_log
    from .s5_attack4_FIT201_template import build_fit201_attack_tail, load_fit201_template
    from .s5_attack4_LIT301_template import build_lit301_attack_tail, load_lit301_template
    from .s5_attack4_MV201_template import build_mv201_attack_tail, load_mv201_template
    from .s5_attack4_P101_template import build_p101_attack_tail, load_p101_template
except ImportError:
    from attack_injection_logging import write_attack_injection_log
    from s5_attack4_FIT201_template import build_fit201_attack_tail, load_fit201_template
    from s5_attack4_LIT301_template import build_lit301_attack_tail, load_lit301_template
    from s5_attack4_MV201_template import build_mv201_attack_tail, load_mv201_template
    from s5_attack4_P101_template import build_p101_attack_tail, load_p101_template

MANIPULATED_PV_COLUMNS = ("MV201.Status", "P101.Status")
DIRECTLY_AFFECTED_PV_COLUMNS = ("FIT201.Pv", "LIT301.Pv")
SUPERVISORY_ATTACK_VIEW_MODE = "main_attack_pv_replaced_with_test_base"
CONTROL_ATTACK_CSV_DIRNAME = "_control_attack_csv"


def _is_status_column(column: str) -> bool:
    return column in MANIPULATED_PV_COLUMNS or column.endswith(".Status")


def sample_injection_starts(
    inject_start: int,
    inject_end: int,
    n_injections: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample distinct start rows from the half-open range ``[inject_start, inject_end)``."""
    if not (0 <= inject_start < inject_end):
        raise ValueError("Need 0 <= inject_start < inject_end.")
    choices = np.arange(inject_start, inject_end, dtype=np.int64)
    if n_injections > len(choices):
        raise ValueError(
            f"Need {n_injections} distinct starts but only {len(choices)} valid positions exist."
        )
    return rng.choice(choices, size=n_injections, replace=False)


def _assign_column_tail(
    df: pd.DataFrame,
    column: str,
    start: int,
    sequence: Sequence[float],
) -> None:
    """Overwrite ``df[column][start:]`` in-place with ``sequence``."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} missing in dataframe.")

    source_series = df[column]
    seq = np.asarray(sequence, dtype=float)
    if _is_status_column(column):
        seq = np.rint(seq)
        if pd.api.types.is_integer_dtype(source_series.dtype):
            values = source_series.to_numpy(copy=True)
            seq = seq.astype(values.dtype, copy=False)
        else:
            values = pd.to_numeric(source_series, errors="coerce").to_numpy(
                dtype=float,
                copy=True,
            )
    else:
        values = pd.to_numeric(source_series, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
    if start < 0 or start > len(values):
        raise IndexError(f"start={start} is outside column length {len(values)}.")
    if len(seq) != len(values) - start:
        raise ValueError(
            f"Tail length mismatch for {column!r}: got {len(seq)} values, "
            f"expected {len(values) - start}."
        )
    values[start:] = seq
    df[column] = values


def export_attack4_dataset(
    df: pd.DataFrame,
    *,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    inject_start: int,
    inject_end: int,
    template_csv_path: str,
    template_start: int,
    template_end: int,
    fit201_attack_progress_span: int,
    lit301_attack_progress_span: int,
    n_injections: int,
    random_seed: int,
    output_dir: str,
    training_filename: str,
    test_filename_prefix: str,
    test_base_filename: str,
    meta_filename: str,
    source_csv_path: Optional[str],
    control_attack_csv_dir: Optional[str] = None,
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """Write attack4 outputs under manipulated/directly-affected PV semantics."""
    os.makedirs(output_dir, exist_ok=True)
    if control_attack_csv_dir is None:
        control_attack_csv_dir = os.path.join(output_dir, CONTROL_ATTACK_CSV_DIRNAME)
    os.makedirs(control_attack_csv_dir, exist_ok=True)
    n = len(df)
    if not (0 <= train_start < train_end <= n):
        raise IndexError(
            f"Invalid training slice: need 0 <= train_start < train_end <= {n}, "
            f"got [{train_start}, {train_end})."
        )
    if not (0 <= test_start < test_end <= n):
        raise IndexError(
            f"Invalid testing slice: need 0 <= test_start < test_end <= {n}, "
            f"got [{test_start}, {test_end})."
        )

    columns_ordered = list(DIRECTLY_AFFECTED_PV_COLUMNS) + list(
        MANIPULATED_PV_COLUMNS
    )
    missing = [c for c in columns_ordered if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing from dataframe: {missing}; have: {list(df.columns)}"
        )

    test_base = df.iloc[test_start:test_end].copy()
    test_len = len(test_base)
    if not (0 <= inject_start < inject_end <= test_len):
        raise IndexError(
            f"Invalid inject-start range: need 0 <= inject_start < inject_end <= {test_len}, "
            f"got [{inject_start}, {inject_end})."
        )

    rng = np.random.default_rng(random_seed)
    starts_rel = sample_injection_starts(inject_start, inject_end, n_injections, rng)

    train_df = df.iloc[train_start:train_end].copy()
    train_df.to_csv(os.path.join(output_dir, training_filename), index=False)
    train_df.to_csv(os.path.join(control_attack_csv_dir, training_filename), index=False)
    test_base.to_csv(os.path.join(output_dir, test_base_filename), index=False)
    test_base.to_csv(os.path.join(control_attack_csv_dir, test_base_filename), index=False)

    fit201_template = load_fit201_template(template_csv_path, template_start, template_end)
    lit301_template = load_lit301_template(template_csv_path, template_start, template_end)
    mv201_template = load_mv201_template(template_csv_path, template_start, template_end)
    p101_template = load_p101_template(template_csv_path, template_start, template_end)

    points_draw_order: List[Dict[str, Any]] = []
    test_files_list: List[Dict[str, Any]] = []

    for i, s_rel in enumerate(starts_rel.tolist()):
        s_rel = int(s_rel)
        s_abs = test_start + s_rel
        tail_len = test_len - s_rel
        fit201_start_value = float(test_base["FIT201.Pv"].iloc[s_rel])
        lit301_start_value = float(test_base["LIT301.Pv"].iloc[s_rel])

        sequences = {
            "LIT301.Pv": build_lit301_attack_tail(
                csv_path=template_csv_path,
                start_index=template_start,
                end_index=template_end,
                injection_length=tail_len,
                start_value=lit301_start_value,
                attack_progress_span=lit301_attack_progress_span,
            ),
            "FIT201.Pv": build_fit201_attack_tail(
                csv_path=template_csv_path,
                start_index=template_start,
                end_index=template_end,
                injection_length=tail_len,
                start_value=fit201_start_value,
                attack_progress_span=fit201_attack_progress_span,
            ),
            "MV201.Status": build_mv201_attack_tail(
                csv_path=template_csv_path,
                start_index=template_start,
                end_index=template_end,
                injection_length=tail_len,
            ),
            "P101.Status": build_p101_attack_tail(
                csv_path=template_csv_path,
                start_index=template_start,
                end_index=template_end,
                injection_length=tail_len,
            ),
        }

        control_tdf = test_base.copy()
        per_column: Dict[str, Any] = {}
        for col in columns_ordered:
            _assign_column_tail(control_tdf, col, s_rel, sequences[col])
            row_meta = {
                "injection_start_row_in_test_file": s_rel,
                "injection_end_row_in_test_file_exclusive": test_len,
                "injection_start_row_in_source": s_abs,
                "injection_end_row_in_source_exclusive": test_end,
                "actual_injected_length": tail_len,
            }
            if col == "FIT201.Pv":
                row_meta["entry_start_value_from_clean_test"] = fit201_start_value
                row_meta["entry_attack_progress_span"] = int(
                    fit201_attack_progress_span
                )
            elif col == "LIT301.Pv":
                row_meta["entry_start_value_from_clean_test"] = lit301_start_value
                row_meta["entry_attack_progress_span"] = int(
                    lit301_attack_progress_span
                )
            elif col in MANIPULATED_PV_COLUMNS:
                row_meta["supervisory_view_value_source"] = test_base_filename
                row_meta["control_attack_csv_value_source"] = "attack_template"
            per_column[col] = row_meta

        fname = f"{test_filename_prefix}{i:02d}.csv"
        supervisory_tdf = control_tdf.copy()
        for col in MANIPULATED_PV_COLUMNS:
            supervisory_tdf[col] = test_base[col].to_numpy(copy=False)
        supervisory_tdf.to_csv(os.path.join(output_dir, fname), index=False)
        control_tdf.to_csv(os.path.join(control_attack_csv_dir, fname), index=False)

        points_draw_order.append(
            {
                "attack_point_id": i,
                "shared_injection_start_row_in_test_file": s_rel,
                "shared_injection_end_row_in_test_file_exclusive": test_len,
                "shared_injection_start_row_in_source": s_abs,
                "shared_injection_end_row_in_source_exclusive": test_end,
                "per_column": per_column,
            }
        )
        test_files_list.append(
            {
                "filename": fname,
                "index": i,
                "attack_point_id": i,
                "control_attack_csv": os.path.join(CONTROL_ATTACK_CSV_DIRNAME, fname),
                "note": (
                    "One coordinated injection per file. In the supervisory-view test CSV, "
                    "manipulated PVs stay at clean test_base values while affected PVs "
                    "follow attack-response values. The sibling control attack CSV keeps "
                    "manipulated PVs at attack values for PLC-side packet injection."
                ),
            }
        )

    points_sorted = sorted(
        points_draw_order,
        key=lambda x: x["shared_injection_start_row_in_test_file"],
    )

    attack_injection_log = write_attack_injection_log(
        output_dir=output_dir,
        attack_name="attack4",
        source_df=df,
        timestamp_col=timestamp_col,
        training_filename=training_filename,
        test_base_filename=test_base_filename,
        manipulated_pv_columns=MANIPULATED_PV_COLUMNS,
        directly_affected_pv_columns=DIRECTLY_AFFECTED_PV_COLUMNS,
        attack_points_in_draw_order=points_draw_order,
        test_files_summary=test_files_list,
    )
    control_attack_injection_log = write_attack_injection_log(
        output_dir=control_attack_csv_dir,
        attack_name="attack4",
        source_df=df,
        timestamp_col=timestamp_col,
        training_filename=training_filename,
        test_base_filename=test_base_filename,
        manipulated_pv_columns=MANIPULATED_PV_COLUMNS,
        directly_affected_pv_columns=DIRECTLY_AFFECTED_PV_COLUMNS,
        attack_points_in_draw_order=points_draw_order,
        test_files_summary=test_files_list,
    )

    meta: Dict[str, Any] = {
        "description": (
            "Metadata for s5 supervisory_historian_attack under manipulated-PV "
            "and directly-affected-PV semantics."
        ),
        "data_source": {
            "path": os.path.abspath(source_csv_path) if source_csv_path else None,
            "manipulated_pv_columns": list(MANIPULATED_PV_COLUMNS),
            "directly_affected_pv_columns": list(DIRECTLY_AFFECTED_PV_COLUMNS),
        },
        "output_directory": os.path.abspath(output_dir),
        "supervisory_view": {
            "mode": SUPERVISORY_ATTACK_VIEW_MODE,
            "clean_replaced_columns": list(MANIPULATED_PV_COLUMNS),
            "note": (
                "Public test_XX.csv files are the supervisory/historian view: "
                "manipulated PV columns are copied from test_base, while directly "
                "affected PV columns keep attack-response values."
            ),
        },
        "control_attack_csv": {
            "directory": os.path.abspath(control_attack_csv_dir),
            "dirname": CONTROL_ATTACK_CSV_DIRNAME,
            "note": (
                "Control-layer injection should read these sibling CSVs because "
                "they preserve attack values for manipulated PV columns."
            ),
            "attack_injection_log": control_attack_injection_log,
        },
        "training": {
            "output_file": training_filename,
            "source_row_range_half_open": [train_start, train_end],
            "note": "Slice from source; no injection.",
        },
        "testing": {
            "source_row_range_half_open": [test_start, test_end],
            "rows_per_test_file": test_len,
            "filename_prefix": test_filename_prefix,
            "clean_test_file_no_injection": test_base_filename,
            "note": (
                "test_base is the same test window with no tampering; test_XX files "
                "only differ by the shared injection start."
            ),
        },
        "inject_window": {
            "note": (
                "Shared injection starts are sampled only inside this half-open start "
                "range within each test file. inject_end does not cap any attack tail."
            ),
            "half_open_start_range": [inject_start, inject_end],
        },
        "sampling_and_reproducibility": {
            "method": "Uniform random without replacement over valid start rows only.",
            "eligible_start_min_inclusive": inject_start,
            "eligible_start_max_inclusive": inject_end - 1,
            "n_eligible_start_positions": inject_end - inject_start,
            "n_attack_points_drawn": n_injections,
            "random_seed": random_seed,
            "implementation": "numpy.random.Generator.choice(..., replace=False)",
        },
        "attack_waveform_by_column": {
            "LIT301.Pv": {
                "role": "directly_affected_pv",
                "type": "template_plus_continuation_to_test_end",
                "template_module": "s5_attack4_LIT301_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "LIT 301",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(lit301_template)),
                "entry_transition": "template_progress_shape_from_clean_start_value",
                "entry_start_value_source": (
                    "clean LIT301 value at shared injection start row"
                ),
                "entry_attack_progress_span": int(lit301_attack_progress_span),
                "entry_progress_profile_source": (
                    "first attack_progress_span samples from the template-source "
                    "sequence, including local step noise"
                ),
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
            "FIT201.Pv": {
                "role": "directly_affected_pv",
                "type": "template_plus_continuation_to_test_end",
                "template_module": "s5_attack4_FIT201_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "FIT 201",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(fit201_template)),
                "entry_transition": "blend_from_clean_start_value",
                "entry_start_value_source": (
                    "clean FIT201 value at shared injection start row"
                ),
                "entry_attack_progress_span": int(fit201_attack_progress_span),
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
            "MV201.Status": {
                "role": "manipulated_pv",
                "type": "supervisory_clean_value_and_control_attack_value",
                "template_module": "s5_attack4_MV201_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "MV201",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(mv201_template)),
                "entry_transition": "supervisory_view_uses_test_base; control_csv_template_replacement_begins_at_injection_start",
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
            "P101.Status": {
                "role": "manipulated_pv",
                "type": "supervisory_clean_value_and_control_attack_value",
                "template_module": "s5_attack4_P101_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "P101 Status",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(p101_template)),
                "entry_transition": "supervisory_view_uses_test_base; control_csv_template_replacement_begins_at_injection_start",
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
        },
        "attack_points_in_draw_order": points_draw_order,
        "attack_points_sorted_by_start_in_test_file": points_sorted,
        "test_files_summary": test_files_list,
        "attack_injection_log": attack_injection_log,
    }

    with open(os.path.join(output_dir, meta_filename), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def _sorted_test_csv_paths(output_dir: str, test_filename_prefix: str) -> List[str]:
    """Return ``test_00.csv``, ``test_01.csv``, ... sorted by numeric suffix."""
    paths: List[str] = []
    for name in sorted(os.listdir(output_dir)):
        if name.startswith(test_filename_prefix) and name.endswith(".csv"):
            suffix = name[len(test_filename_prefix) : -4]
            if suffix.isdigit():
                paths.append(os.path.join(output_dir, name))
    paths.sort(key=lambda p: int(os.path.basename(p)[len(test_filename_prefix) : -4]))
    return paths


def plot_all_tests_line_chart(
    output_dir: str,
    columns: Sequence[str],
    *,
    test_filename_prefix: str = "test_",
    baseline_filename: Optional[str] = "test_base.csv",
    save_path: str,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11, 6),
) -> str:
    """Overlay every attacked test CSV and the clean baseline."""
    paths = _sorted_test_csv_paths(output_dir, test_filename_prefix)
    if not paths:
        raise FileNotFoundError(
            f"No test CSVs under {output_dir!r} with prefix {test_filename_prefix!r}."
        )

    dfs = [pd.read_csv(p) for p in paths]
    n_rows = len(dfs[0])
    x = np.arange(n_rows)
    labels = [os.path.basename(p) for p in paths]
    cmap = plt.get_cmap("tab20")

    df_base: Optional[pd.DataFrame] = None
    if baseline_filename:
        base_path = os.path.join(output_dir, baseline_filename)
        if os.path.isfile(base_path):
            df_base = pd.read_csv(base_path)

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=(figsize[0], max(figsize[1], 3.2 * len(columns))),
        sharex=True,
        squeeze=False,
    )
    ax_flat = axes.ravel()
    for ax_idx, col in enumerate(columns):
        ax = ax_flat[ax_idx]
        if df_base is not None:
            ax.plot(
                x,
                df_base[col].to_numpy(dtype=float),
                color="black",
                linestyle="--",
                linewidth=1.7,
                label=f"{baseline_filename} (no attack)" if ax_idx == 0 else None,
            )
        for i, df_attack in enumerate(dfs):
            color = cmap(i / max(len(dfs) - 1, 1) if len(dfs) > 1 else 0)
            ax.plot(
                x,
                df_attack[col].to_numpy(dtype=float),
                color=color,
                linewidth=0.9,
                alpha=0.78,
                label=labels[i] if ax_idx == 0 else None,
            )
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_title(title or "Attack4: coordinated full-tail replacement")
        if ax_idx == len(columns) - 1:
            ax.set_xlabel("Row index in test file (0 = first row)")

    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(5, max(len(labels), 1)),
        fontsize=7,
        frameon=True,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18 if len(labels) <= 8 else 0.22)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    repo_root = _resolve_repo_root()
    input_csv = repo_root / "dataset" / "swat" / "physics" / "Dec2019_dealed.csv"
    template_csv = os.path.join(data_dir, "SWaT_dataset_Jul 19 v2_dealed.csv")
    output_dir = data_dir
    training_filename = "training.csv"
    test_filename_prefix = "test_"
    test_base_filename = "test_base.csv"
    meta_filename = "meta.json"

    train_start, train_end = 1500, 3300  # 3300 corrresponds to 2019/12/6  11:00:00
    test_start, test_end = 3300, 5100
    inject_start, inject_end = 50, 650  # relative to test_start
    n_injections = 20
    random_seed = 45

    template_start, template_end = 11228, 13936
    # template_start, template_end = 11235, 13936
    fit201_attack_progress_span = 18
    lit301_attack_progress_span = 448

    df = pd.read_csv(input_csv)
    meta = export_attack4_dataset(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        inject_start=inject_start,
        inject_end=inject_end,
        template_csv_path=template_csv,
        template_start=template_start,
        template_end=template_end,
        fit201_attack_progress_span=fit201_attack_progress_span,
        lit301_attack_progress_span=lit301_attack_progress_span,
        n_injections=n_injections,
        random_seed=random_seed,
        output_dir=output_dir,
        training_filename=training_filename,
        test_filename_prefix=test_filename_prefix,
        test_base_filename=test_base_filename,
        meta_filename=meta_filename,
        source_csv_path=input_csv,
        timestamp_col="timestamp",
    )

    print(
        f"Wrote {os.path.join(output_dir, training_filename)} "
        f"(rows [{train_start}, {train_end}), no attack)"
    )
    print(
        f"Wrote {os.path.join(output_dir, test_base_filename)} "
        f"(rows [{test_start}, {test_end}), clean)"
    )
    for t in meta["test_files_summary"]:
        pt = meta["attack_points_in_draw_order"][t["index"]]
        print(
            f"Wrote {os.path.join(output_dir, t['filename'])} "
            f"(shared_start={pt['shared_injection_start_row_in_test_file']}, "
            f"shared_end={pt['shared_injection_end_row_in_test_file_exclusive']})"
        )
    print(f"Wrote {os.path.join(output_dir, meta_filename)}")
    print(f"Wrote {meta['attack_injection_log']['csv_path']}")
    print(f"Wrote {meta['attack_injection_log']['json_path']}")
    print(f"Wrote control attack CSVs under {meta['control_attack_csv']['directory']}")
    print(f"Wrote {meta['control_attack_csv']['attack_injection_log']['csv_path']}")
    print(f"Wrote {meta['control_attack_csv']['attack_injection_log']['json_path']}")

    plot_path = os.path.join(output_dir, "all_tests_line_chart.png")
    saved = plot_all_tests_line_chart(
        output_dir,
        ("LIT301.Pv", "FIT201.Pv", "MV201.Status", "P101.Status"),
        test_filename_prefix=test_filename_prefix,
        baseline_filename=test_base_filename,
        save_path=plot_path,
        title="Attack4 S4/S5 supervisory view: test_base (dashed) vs test_XX",
    )
    print(f"Wrote {saved}")

    control_attack_csv_dir = meta["control_attack_csv"]["directory"]
    control_plot_path = os.path.join(control_attack_csv_dir, "all_tests_line_chart.png")
    control_saved = plot_all_tests_line_chart(
        control_attack_csv_dir,
        ("LIT301.Pv", "FIT201.Pv", "MV201.Status", "P101.Status"),
        test_filename_prefix=test_filename_prefix,
        baseline_filename=test_base_filename,
        save_path=control_plot_path,
        title="Attack4 S4/S5 control source: test_base (dashed) vs _control_attack_csv/test_XX",
    )
    print(f"Wrote {control_saved}")


if __name__ == "__main__":
    main()
