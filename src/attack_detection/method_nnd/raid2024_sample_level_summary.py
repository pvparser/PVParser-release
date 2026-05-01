"""Build sample-level summaries from an existing RAID 2024 reproduction run.

The main `raid2024_reproduction.py` pipeline already evaluates models on
sample-level counts internally, but its stage summary JSON files are currently
organized around the repository's custom window/pcap-centric reporting.

This helper reuses the saved `summary.json` and each perspective's
`file_summary_csv` to materialize summaries that are much closer to the
original paper's sample-level evaluation:

* validation confusion matrix on samples
* precision / recall / F1 / accuracy
* tacc / vacc / bacc
* attack-instance TTE statistics

It does not retrain any model. It only post-processes an existing run folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
import re
from typing import Any, Iterable, Optional


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "attack_detection").is_dir() and (parent / "src" / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing src/attack_detection.")


REPO_ROOT = _repo_root()
METHOD_NND_ROOT = REPO_ROOT / "src" / "attack_detection" / "method_nnd"
RESULTS_ROOT = METHOD_NND_ROOT / "results"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def _split_sort_key(split_name: str) -> tuple[int, Any]:
    if split_name == "test_base":
        return (0, -1)
    match = re.fullmatch(r"test_(\d+)", split_name)
    if match:
        return (1, int(match.group(1)))
    return (2, split_name)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _confusion_metrics_dict(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    total = tp + fp + fn + tn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    tnr = _safe_ratio(tn, tn + fp)
    fpr = _safe_ratio(fp, fp + tn)
    return {
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "precision": precision,
        "recall": recall,
        "tnr": tnr,
        "fpr": fpr,
        "f1": _binary_f1(tp, fp, fn),
        "accuracy": _safe_ratio(tp + tn, total),
    }


def _tte_stats(tte_values: Iterable[float]) -> dict[str, Any]:
    values = list(float(value) for value in tte_values)
    if not values:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _sample_metrics_with_balanced_accuracy(
    *,
    train_true_normal: int,
    train_false_alarm: int,
    sample_tp: int,
    sample_fp: int,
    sample_fn: int,
    sample_tn: int,
) -> dict[str, Any]:
    tacc = _safe_ratio(train_true_normal, train_true_normal + train_false_alarm)
    vacc = 0.5 * (
        _safe_ratio(sample_tn, sample_tn + sample_fp) + _safe_ratio(sample_tp, sample_tp + sample_fn)
    )
    bacc = 0.5 * (tacc + vacc)
    return {
        "tacc": tacc,
        "vacc": vacc,
        "bacc": bacc,
        "train_true_normal": train_true_normal,
        "train_false_alarm": train_false_alarm,
        **_confusion_metrics_dict(sample_tp, sample_fp, sample_fn, sample_tn),
    }


def _summarize_task_sample_level(
    *,
    task_summary: dict[str, Any],
    perspective: str,
) -> Optional[dict[str, Any]]:
    perspective_summary = task_summary.get("perspectives", {}).get(perspective)
    if not perspective_summary:
        return None

    file_summary_csv = perspective_summary.get("file_summary_csv")
    best_candidate = perspective_summary.get("best_candidate")
    if not file_summary_csv or not best_candidate:
        return None

    rows = _load_csv_rows(Path(file_summary_csv))
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["split_name"]), []).append(row)

    tests: list[dict[str, Any]] = []
    tte_values: list[float] = []
    attack_test_count = 0
    clean_test_count = 0

    sample_tp = 0
    sample_fp = 0
    sample_fn = 0
    sample_tn = 0
    attack_sample_count = 0
    normal_sample_count = 0
    valid_sample_count = 0

    for split_name in sorted(grouped_rows, key=_split_sort_key):
        split_rows = grouped_rows[split_name]
        label = int(split_rows[0]["label"])
        attack_name = split_rows[0].get("attack_name")
        split_sample_tp = sum(int(row.get("sample_tp_count", 0) or 0) for row in split_rows)
        split_sample_fp = sum(int(row.get("sample_fp_count", 0) or 0) for row in split_rows)
        split_sample_fn = sum(int(row.get("sample_fn_count", 0) or 0) for row in split_rows)
        split_sample_tn = sum(int(row.get("sample_tn_count", 0) or 0) for row in split_rows)
        split_attack_sample_count = sum(int(row.get("attack_sample_count", 0) or 0) for row in split_rows)
        split_normal_sample_count = sum(int(row.get("normal_sample_count", 0) or 0) for row in split_rows)
        split_sample_count = sum(int(row.get("sample_count", 0) or 0) for row in split_rows)
        split_tte_candidates = [
            float(row["time_to_exposure_seconds"])
            for row in split_rows
            if row.get("time_to_exposure_seconds") not in (None, "")
        ]
        split_tte = min(split_tte_candidates) if split_tte_candidates else None
        if split_tte is not None:
            tte_values.append(split_tte)

        sample_tp += split_sample_tp
        sample_fp += split_sample_fp
        sample_fn += split_sample_fn
        sample_tn += split_sample_tn
        attack_sample_count += split_attack_sample_count
        normal_sample_count += split_normal_sample_count
        valid_sample_count += split_sample_count

        if label == 1:
            attack_test_count += 1
        else:
            clean_test_count += 1

        tests.append(
            {
                "task_id": str(task_summary["task_id"]),
                "stage": str(task_summary["stage"]),
                "namespace": str(task_summary["namespace"]),
                "perspective": perspective,
                "split_name": split_name,
                "label": "attack" if label == 1 else "clean",
                "attack_name": attack_name,
                "sample_count": split_sample_count,
                "attack_sample_count": split_attack_sample_count,
                "normal_sample_count": split_normal_sample_count,
                "tte_seconds": split_tte,
                **_confusion_metrics_dict(
                    split_sample_tp,
                    split_sample_fp,
                    split_sample_fn,
                    split_sample_tn,
                ),
            }
        )

    metrics = _sample_metrics_with_balanced_accuracy(
        train_true_normal=int(best_candidate.get("train_true_normal", 0) or 0),
        train_false_alarm=int(best_candidate.get("train_false_alarm", 0) or 0),
        sample_tp=sample_tp,
        sample_fp=sample_fp,
        sample_fn=sample_fn,
        sample_tn=sample_tn,
    )

    return {
        "task_id": str(task_summary["task_id"]),
        "stage": str(task_summary["stage"]),
        "namespace": str(task_summary["namespace"]),
        "perspective": perspective,
        "algorithm": best_candidate.get("algorithm"),
        "params": best_candidate.get("params"),
        "train_sample_count": int(best_candidate.get("train_samples_after_subsampling", best_candidate.get("train_samples", 0)) or 0),
        "valid_sample_count": valid_sample_count,
        "attack_sample_count": attack_sample_count,
        "normal_sample_count": normal_sample_count,
        "attack_test_count": attack_test_count,
        "clean_test_count": clean_test_count,
        "tte": _tte_stats(tte_values),
        "tests": tests,
        **metrics,
    }


def _aggregate_stage_task_rollups(task_rollups: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not task_rollups:
        return None

    stage = str(task_rollups[0]["stage"])
    perspective = str(task_rollups[0]["perspective"])

    tests: list[dict[str, Any]] = []
    task_ids: list[str] = []
    namespaces: list[str] = []
    algorithms: list[str] = []
    train_true_normal = 0
    train_false_alarm = 0
    sample_tp = 0
    sample_fp = 0
    sample_fn = 0
    sample_tn = 0
    valid_sample_count = 0
    attack_sample_count = 0
    normal_sample_count = 0
    attack_test_count = 0
    clean_test_count = 0
    tte_values: list[float] = []

    for task_rollup in task_rollups:
        task_ids.append(str(task_rollup["task_id"]))
        namespaces.append(str(task_rollup["namespace"]))
        if task_rollup.get("algorithm"):
            algorithms.append(str(task_rollup["algorithm"]))
        train_true_normal += int(task_rollup.get("train_true_normal", 0) or 0)
        train_false_alarm += int(task_rollup.get("train_false_alarm", 0) or 0)
        confusion = task_rollup["confusion_matrix"]
        sample_tp += int(confusion["tp"])
        sample_fp += int(confusion["fp"])
        sample_fn += int(confusion["fn"])
        sample_tn += int(confusion["tn"])
        valid_sample_count += int(task_rollup.get("valid_sample_count", 0) or 0)
        attack_sample_count += int(task_rollup.get("attack_sample_count", 0) or 0)
        normal_sample_count += int(task_rollup.get("normal_sample_count", 0) or 0)
        attack_test_count += int(task_rollup.get("attack_test_count", 0) or 0)
        clean_test_count += int(task_rollup.get("clean_test_count", 0) or 0)
        tests.extend(task_rollup["tests"])

        tte = task_rollup.get("tte", {})
        if tte.get("count", 0):
            for test in task_rollup["tests"]:
                if test.get("tte_seconds") is not None:
                    tte_values.append(float(test["tte_seconds"]))

    tests.sort(key=lambda item: (str(item["namespace"]), _split_sort_key(str(item["split_name"]))))
    overview = _sample_metrics_with_balanced_accuracy(
        train_true_normal=train_true_normal,
        train_false_alarm=train_false_alarm,
        sample_tp=sample_tp,
        sample_fp=sample_fp,
        sample_fn=sample_fn,
        sample_tn=sample_tn,
    )

    return {
        "stage": stage,
        "perspective": perspective,
        "task_ids": task_ids,
        "task_count": len(task_rollups),
        "namespaces": namespaces,
        "algorithms": algorithms,
        "train_true_normal": train_true_normal,
        "train_false_alarm": train_false_alarm,
        "valid_sample_count": valid_sample_count,
        "attack_sample_count": attack_sample_count,
        "normal_sample_count": normal_sample_count,
        "attack_test_count": attack_test_count,
        "clean_test_count": clean_test_count,
        "overview": {
            **overview,
            "tte": _tte_stats(tte_values),
        },
        "tasks": task_rollups,
        "tests": tests,
        "confusion_matrix": overview["confusion_matrix"],
        "precision": overview["precision"],
        "recall": overview["recall"],
        "f1": overview["f1"],
        "accuracy": overview["accuracy"],
        "tacc": overview["tacc"],
        "vacc": overview["vacc"],
        "bacc": overview["bacc"],
        "tte_count": len(tte_values),
        "tte_mean_seconds": _tte_stats(tte_values)["mean_seconds"],
        "tte_median_seconds": _tte_stats(tte_values)["median_seconds"],
        "tte_min_seconds": _tte_stats(tte_values)["min_seconds"],
        "tte_max_seconds": _tte_stats(tte_values)["max_seconds"],
    }


def _latest_result_dir() -> Path:
    candidates = sorted(
        [path for path in RESULTS_ROOT.glob("raid2024_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No RAID 2024 result directories found under {RESULTS_ROOT}")
    return candidates[-1]


def build_sample_level_summaries(
    *,
    result_dir: Path,
) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary.json under {result_dir}")

    summary = json.loads(summary_path.read_text())
    tasks = list(summary.get("tasks", []))
    perspectives = list(summary.get("perspectives", []))

    stage_perspective_rollups: dict[str, dict[str, Any]] = {}
    overview_rows: list[dict[str, Any]] = []

    for perspective in perspectives:
        grouped_by_stage: dict[str, list[dict[str, Any]]] = {}
        for task_summary in tasks:
            task_rollup = _summarize_task_sample_level(task_summary=task_summary, perspective=perspective)
            if task_rollup is None:
                continue
            grouped_by_stage.setdefault(str(task_summary["stage"]), []).append(task_rollup)

        stage_rollups: dict[str, Any] = {}
        for stage, task_rollups in grouped_by_stage.items():
            rollup = _aggregate_stage_task_rollups(task_rollups)
            if rollup is None:
                continue
            stage_rollups[stage] = rollup
            confusion = rollup["confusion_matrix"]
            overview_rows.append(
                {
                    "stage": stage,
                    "perspective": perspective,
                    "tp": confusion["tp"],
                    "fp": confusion["fp"],
                    "fn": confusion["fn"],
                    "tn": confusion["tn"],
                    "precision": rollup["precision"],
                    "recall": rollup["recall"],
                    "f1": rollup["f1"],
                    "accuracy": rollup["accuracy"],
                    "tacc": rollup["tacc"],
                    "vacc": rollup["vacc"],
                    "bacc": rollup["bacc"],
                    "tte_mean_seconds": rollup["tte_mean_seconds"],
                    "tte_median_seconds": rollup["tte_median_seconds"],
                }
            )

        if stage_rollups:
            stage_perspective_rollups[perspective] = stage_rollups

    written_files: dict[str, str] = {}
    for perspective, stage_rollups in stage_perspective_rollups.items():
        for stage, rollup in stage_rollups.items():
            out_path = result_dir / f"{stage}__{perspective}__sample_level_summary.json"
            out_path.write_text(json.dumps(rollup, indent=2))
            written_files[f"{stage}__{perspective}"] = str(out_path.resolve())

    overview_rows.sort(key=lambda item: (str(item["stage"]), str(item["perspective"])))
    overview_csv = result_dir / "sample_level_overview.csv"
    with overview_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "perspective",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "tacc",
                "vacc",
                "bacc",
                "tte_mean_seconds",
                "tte_median_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(overview_rows)

    sample_level_summary = {
        "source_summary_json": str(summary_path.resolve()),
        "result_dir": str(result_dir.resolve()),
        "evaluation_level": "sample",
        "stage_perspective_rollups": stage_perspective_rollups,
        "written_files": written_files,
        "overview_csv": str(overview_csv.resolve()),
    }
    sample_level_summary_path = result_dir / "sample_level_summary.json"
    sample_level_summary_path.write_text(json.dumps(sample_level_summary, indent=2))
    return sample_level_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build sample-level stage summaries from an existing RAID 2024 reproduction run.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help=(
            "Existing method_nnd result directory containing summary.json. "
            "Defaults to the latest raid2024_* directory under method_nnd/results."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve() if args.result_dir else _latest_result_dir().resolve()
    summary = build_sample_level_summaries(result_dir=result_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
