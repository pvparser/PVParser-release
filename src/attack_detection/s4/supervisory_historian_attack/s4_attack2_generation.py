"""
Build outputs directly under ``../data/`` for the s4 supervisory-historian script.

Program semantics:
- ``training.csv``: rows ``[train_start, train_end)``, no injection.
- ``test_base.csv``: rows ``[test_start, test_end)``, no injection.
- ``test_XX.csv``: supervisory-view attack files; one shared injection point
  per file.
- ``_control_attack_csv/test_XX.csv``: control-layer attack source files with
  the manipulated PV kept at its attack value.
- The shared injection point is sampled only from ``[inject_start, inject_end)``
  inside the test slice.
- ``LIT301.Pv`` is the manipulated PV. In supervisory-view ``test_XX.csv`` it
  remains the clean ``test_base`` value; in ``_control_attack_csv/test_XX.csv``
  it is injected directly from the chosen attack prototype start.
- ``MV201.Status``, ``P101.Status``, and ``FIT301.Pv`` are
  directly affected PVs and are replayed from the same attack prototype start,
  so their natural response delay is preserved by the template itself in both
  views.
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
    from .s4_attack2_FIT301_template import (
        FIT301_LEARNED_RISE_STAGE,
        build_fit301_attack_tail,
        load_fit301_template,
    )
    from .s4_attack2_LIT301_template import (
        LIT301_ATTACK_END_OFFSET,
        LIT301_REFERENCE_ATTACKED_VALUE,
        LIT301_REFERENCE_PRE_ATTACK_VALUE,
        build_lit301_attack_tail,
        load_lit301_template,
    )
    from .s4_attack2_MV201_template import build_mv201_attack_tail, load_mv201_template
    from .s4_attack2_P101_template import build_p101_attack_tail, load_p101_template
except ImportError:
    from attack_injection_logging import write_attack_injection_log
    from s4_attack2_FIT301_template import (
        FIT301_LEARNED_RISE_STAGE,
        build_fit301_attack_tail,
        load_fit301_template,
    )
    from s4_attack2_LIT301_template import (
        LIT301_ATTACK_END_OFFSET,
        LIT301_REFERENCE_ATTACKED_VALUE,
        LIT301_REFERENCE_PRE_ATTACK_VALUE,
        build_lit301_attack_tail,
        load_lit301_template,
    )
    from s4_attack2_MV201_template import build_mv201_attack_tail, load_mv201_template
    from s4_attack2_P101_template import build_p101_attack_tail, load_p101_template

MANIPULATED_PV_COLUMNS = ("LIT301.Pv",)
DIRECTLY_AFFECTED_PV_COLUMNS = (
    "MV201.Status",
    "P101.Status",
    "FIT301.Pv",
)
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


def export_attack2_dataset(
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
    lit301_attack_end_offset: int,
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
    """Write attack2 outputs under manipulated/directly-affected PV semantics."""
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

    columns_ordered = list(MANIPULATED_PV_COLUMNS) + list(DIRECTLY_AFFECTED_PV_COLUMNS)
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

    lit301_template = load_lit301_template(template_csv_path, template_start, template_end)
    mv201_template = load_mv201_template(template_csv_path, template_start, template_end)
    p101_template = load_p101_template(template_csv_path, template_start, template_end)
    fit301_template = load_fit301_template(template_csv_path, template_start, template_end)

    points_draw_order: List[Dict[str, Any]] = []
    test_files_list: List[Dict[str, Any]] = []

    for i, s_rel in enumerate(starts_rel.tolist()):
        s_rel = int(s_rel)
        s_abs = test_start + s_rel
        tail_len = test_len - s_rel
        lit301_pre_attack_value = float(test_base["LIT301.Pv"].iloc[s_rel])
        fit301_entry_value = float(test_base["FIT301.Pv"].iloc[s_rel])

        sequences = {
            "LIT301.Pv": build_lit301_attack_tail(
                csv_path=template_csv_path,
                start_index=template_start,
                end_index=template_end,
                injection_length=tail_len,
                pre_attack_value=lit301_pre_attack_value,
                attack_end_offset=lit301_attack_end_offset,
                reference_pre_attack_value=LIT301_REFERENCE_PRE_ATTACK_VALUE,
                reference_attacked_value=LIT301_REFERENCE_ATTACKED_VALUE,
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
            "FIT301.Pv": build_fit301_attack_tail(
                csv_path=template_csv_path,
                start_index=template_start,
                end_index=template_end,
                injection_length=tail_len,
                learned_rise_stage=FIT301_LEARNED_RISE_STAGE,
                entry_value=fit301_entry_value,
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
            if col == "LIT301.Pv":
                row_meta["reference_pre_attack_value_from_clean_test"] = lit301_pre_attack_value
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
                    "LIT301 stays at clean test_base values while affected PVs replay the "
                    "attack prototype response. The sibling control attack CSV keeps "
                    "LIT301 at the attack value for PLC-side packet injection."
                ),
            }
        )

    points_sorted = sorted(
        points_draw_order,
        key=lambda x: x["shared_injection_start_row_in_test_file"],
    )

    attack_injection_log = write_attack_injection_log(
        output_dir=output_dir,
        attack_name="attack2",
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
        attack_name="attack2",
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
            "Metadata for s4 supervisory_historian_attack under manipulated-PV "
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
                "role": "manipulated_pv",
                "type": "supervisory_clean_value_and_control_attack_value",
                "template_module": "s4_attack2_LIT301_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "LIT 301",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(lit301_template)),
                "entry_transition": (
                    "supervisory_view_uses_test_base; control_csv_uses_strict_template_attack_segment_then_shifted_post_attack_recovery"
                ),
                "attack_end_offset_in_template": int(lit301_attack_end_offset),
                "reference_pre_attack_value_in_prototype": LIT301_REFERENCE_PRE_ATTACK_VALUE,
                "reference_first_attacked_value_in_prototype": LIT301_REFERENCE_ATTACKED_VALUE,
                "reanchoring_rule": (
                    "Keep template values unchanged before attack_end_offset_in_template. "
                    "From attack_end_offset_in_template onward, shift the template-source "
                    "recovery segment by clean_pre_attack_value - "
                    "reference_pre_attack_value_in_prototype."
                ),
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
            "MV201.Status": {
                "role": "directly_affected_pv",
                "type": "template_plus_continuation_to_test_end",
                "template_module": "s4_attack2_MV201_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "MV201",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(mv201_template)),
                "entry_transition": "replay_from_attack_prototype_start",
                "natural_response_note": (
                    "This discrete PV stays normal until the prototype itself changes, "
                    "so the response delay is encoded by the template."
                ),
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
            "P101.Status": {
                "role": "directly_affected_pv",
                "type": "template_plus_continuation_to_test_end",
                "template_module": "s4_attack2_P101_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "P101 Status",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(p101_template)),
                "entry_transition": "replay_from_attack_prototype_start",
                "natural_response_note": (
                    "This discrete PV stays normal until the prototype itself changes, "
                    "so the response delay is encoded by the template."
                ),
                "actual_injection_length_rule": (
                    "test_file_length - shared_injection_start_row_in_test_file"
                ),
                "tail_behavior_after_source_continuation_exhaustion": (
                    "Hold the final available template-source value."
                ),
            },
            "FIT301.Pv": {
                "role": "directly_affected_pv",
                "type": "template_plus_continuation_to_test_end",
                "template_module": "s4_attack2_FIT301_template.py",
                "template_csv": os.path.abspath(template_csv_path),
                "template_column": "FIT 301",
                "template_slice_in_source_half_open": [template_start, template_end],
                "template_length": int(len(fit301_template)),
                "entry_transition": "replay_from_attack_prototype_start",
                "natural_response_note": (
                    "The generated tail preserves the original attack2 template and "
                    "continuation after the entry point. A short bridge is only used "
                    "near the injection onset so low-state starts do not jump directly "
                    "to the high plateau."
                ),
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
    figsize: Tuple[float, float] = (11, 7),
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
            ax.set_title(title or "Attack2: coordinated full-tail replacement")
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

    train_start, train_end = 1500, 3300
    test_start, test_end = 3300, 5100
    inject_start, inject_end = 50, 650  # relative to test_start
    n_injections = 20
    random_seed = 44
    template_start, template_end = 9803, 12511
    lit301_attack_end_offset = LIT301_ATTACK_END_OFFSET

    df = pd.read_csv(input_csv)
    meta = export_attack2_dataset(
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
        lit301_attack_end_offset=lit301_attack_end_offset,
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
        (
            "LIT301.Pv",
            "MV201.Status",
            "P101.Status",
            "FIT301.Pv",
        ),
        test_filename_prefix=test_filename_prefix,
        baseline_filename=test_base_filename,
        save_path=plot_path,
        title="Attack2 S4/S5 supervisory view: test_base (dashed) vs test_XX",
    )
    print(f"Wrote {saved}")

    control_attack_csv_dir = meta["control_attack_csv"]["directory"]
    control_plot_path = os.path.join(control_attack_csv_dir, "all_tests_line_chart.png")
    control_saved = plot_all_tests_line_chart(
        control_attack_csv_dir,
        (
            "LIT301.Pv",
            "MV201.Status",
            "P101.Status",
            "FIT301.Pv",
        ),
        test_filename_prefix=test_filename_prefix,
        baseline_filename=test_base_filename,
        save_path=control_plot_path,
        title="Attack2 S4/S5 control source: test_base (dashed) vs _control_attack_csv/test_XX",
    )
    print(f"Wrote {control_saved}")


if __name__ == "__main__":
    main()
