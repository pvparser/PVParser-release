#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent.parent
GECO_CODE_ROOT = REPO_ROOT / "src" / "attack_detection" / "method_geco" / "artifact" / "code" / "ipal-ids-framework"

if str(GECO_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GECO_CODE_ROOT))


def load_pasad_scenario_module():
    script_path = SCRIPT_ROOT / "pasad_scenario.py"
    spec = importlib.util.spec_from_file_location("pasad_scenario", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pasad_scenario = load_pasad_scenario_module()

try:
    import ipal_iids.settings as geco_settings  # noqa: E402
    from ids.GeCo.GeCo import GeCo  # noqa: E402
except ImportError as exc:  # pragma: no cover - GeCo is optional for PASAD-only tables.
    geco_settings = None
    GeCo = None
    GECO_IMPORT_ERROR = exc
else:
    GECO_IMPORT_ERROR = None


@dataclass
class InstanceResult:
    test_name: str
    attack_start_timestamp: str
    attack_end_timestamp_exclusive: str
    pre_attack_detected: bool
    attack_detected: bool
    first_pre_attack_alarm_timestamp: str | None
    first_in_attack_alarm_timestamp: str | None
    tte_seconds: float | None


@dataclass
class EvaluationSummary:
    scenario: str
    description: str
    method_id: str
    view: str
    channels: list[str]
    max_formel_length: int
    threshold_factor: float
    cusum_factor: float
    cpus: int
    control_timestamp_shift_seconds: int | None
    n_instances: int
    total_phases: int
    tp: int
    fp: int
    tn: int
    fn: int
    acc: float
    f1: float
    aer: float
    strict_tp_instances: int
    strict_aer: float
    mean_tte_seconds: float | None
    fn_30s: int
    fn_60s: int
    clean_test_base_alarm_rows: int
    instances: list[InstanceResult]


def json_number(value: Any) -> int | float:
    raw = float(value)
    rounded = round(raw)
    if abs(raw - rounded) < 1e-9:
        return int(rounded)
    return raw


def build_state_line(
    *,
    row_id: int,
    timestamp_epoch: float,
    state: dict[str, Any],
    malicious: bool,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "timestamp": int(round(float(timestamp_epoch))),
        "state": {key: json_number(value) for key, value in state.items()},
        "malicious": bool(malicious),
    }


def write_state_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_state_rows_for_split(
    *,
    config,
    view: str,
    split_name: str,
    channels: tuple[str, ...],
    control_timestamp_shift_seconds: int,
) -> list[dict[str, Any]]:
    attack_log = pasad_scenario.load_attack_log(config)
    attack_meta = {
        row["output_file_stem"]: pasad_scenario.attack_window_epoch(row)
        for _, row in attack_log.iterrows()
    }
    df = pasad_scenario.load_split(
        config=config,
        view=view,
        split_name=split_name,
        channels=channels,
        control_timestamp_shift_seconds=control_timestamp_shift_seconds,
    )
    rows: list[dict[str, Any]] = []
    if split_name in attack_meta:
        attack_start_epoch, attack_end_epoch = attack_meta[split_name]
    else:
        attack_start_epoch, attack_end_epoch = None, None

    for row_id, row in df.reset_index(drop=True).iterrows():
        epoch = float(row["timestamp_epoch"])
        malicious = False if attack_start_epoch is None else attack_start_epoch <= epoch < attack_end_epoch
        state = {channel: row[channel] for channel in channels}
        rows.append(
            build_state_line(
                row_id=row_id,
                timestamp_epoch=epoch,
                state=state,
                malicious=malicious,
            )
        )
    return rows


def setup_geco_config(
    *,
    config_path: Path,
    model_file_name: str,
    max_formel_length: int,
    threshold_factor: float,
    cusum_factor: float,
    cpus: int,
) -> dict[str, Any]:
    payload = {
        "GeCo": {
            "_type": "GeCo",
            "model-file": f"./{model_file_name}",
            "ignore": [],
            "max_formel_length": max_formel_length,
            "threshold_factor": threshold_factor,
            "cusum_factor": cusum_factor,
            "cpus": cpus,
        }
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return payload


def train_geco(
    *,
    config_path: Path,
    config_payload: dict[str, Any],
    train_state_path: Path,
) -> GeCo:
    if GeCo is None or geco_settings is None:
        raise ImportError(
            "GeCo dependencies are not available in this checkout. "
            f"Original import error: {GECO_IMPORT_ERROR}"
        )
    geco_settings.config = str(config_path)
    geco_settings.idss = config_payload
    geco_settings.combiner = {"_type": "Any", "model-file": None}
    geco_settings.logger = logging.getLogger("geco-scenario")
    geco = GeCo(name="GeCo")
    geco.train(state=str(train_state_path))
    geco.save_trained_model()
    return geco


def run_live_on_state_file(
    *,
    geco: GeCo,
    input_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emitted: list[dict[str, Any]] = []
    with gzip.open(input_path, "rt") as fin, gzip.open(output_path, "wt") as fout:
        for raw_line in fin:
            msg = json.loads(raw_line)
            alert, score = geco.new_state_msg(msg)
            enriched = {
                "id": msg["id"],
                "timestamp": msg["timestamp"],
                "malicious": msg["malicious"],
                "alerts": {"GeCo": bool(alert)},
                "scores": {"GeCo": score, "Any": int(bool(alert))},
                "ids": bool(alert),
            }
            fout.write(json.dumps(enriched, ensure_ascii=True) + "\n")
            emitted.append(enriched)
    return emitted


def evaluate_outputs(*, config, output_by_test: dict[str, list[dict[str, Any]]]) -> EvaluationSummary:
    attack_log = pasad_scenario.load_attack_log(config)
    instances: list[InstanceResult] = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    strict_tp = 0
    ttes: list[float] = []
    fn_30s = 0
    fn_60s = 0

    for _, row in attack_log.iterrows():
        test_name = row["output_file_stem"]
        attack_start_epoch, attack_end_epoch = pasad_scenario.attack_window_epoch(row)
        alarms = output_by_test[test_name]
        pre_epochs = [item["timestamp"] for item in alarms if item["ids"] and item["timestamp"] < attack_start_epoch]
        in_epochs = [item["timestamp"] for item in alarms if item["ids"] and attack_start_epoch <= item["timestamp"] < attack_end_epoch]

        pre_detected = len(pre_epochs) > 0
        attack_detected = len(in_epochs) > 0

        if pre_detected:
            fp += 1
        else:
            tn += 1

        if attack_detected:
            tp += 1
            tte = float(min(in_epochs) - attack_start_epoch)
            ttes.append(tte)
            if tte > 30:
                fn_30s += 1
            if tte > 60:
                fn_60s += 1
        else:
            fn += 1
            fn_30s += 1
            fn_60s += 1
            tte = None

        if attack_detected and not pre_detected:
            strict_tp += 1

        instances.append(
            InstanceResult(
                test_name=test_name,
                attack_start_timestamp=row["attack_start_timestamp"],
                attack_end_timestamp_exclusive=row["attack_end_timestamp_exclusive"],
                pre_attack_detected=pre_detected,
                attack_detected=attack_detected,
                first_pre_attack_alarm_timestamp=pasad_scenario.format_timestamp_epoch(float(min(pre_epochs))) if pre_epochs else None,
                first_in_attack_alarm_timestamp=pasad_scenario.format_timestamp_epoch(float(min(in_epochs))) if in_epochs else None,
                tte_seconds=tte,
            )
        )

    total = tp + fp + tn + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return EvaluationSummary(
        scenario=config.scenario,
        description=config.description,
        method_id="",
        view="",
        channels=[],
        max_formel_length=0,
        threshold_factor=0.0,
        cusum_factor=0.0,
        cpus=0,
        control_timestamp_shift_seconds=None,
        n_instances=len(instances),
        total_phases=2 * len(instances),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        acc=(tp + tn) / total,
        f1=f1,
        aer=tp / len(instances),
        strict_tp_instances=strict_tp,
        strict_aer=strict_tp / len(instances),
        mean_tte_seconds=None if not ttes else sum(ttes) / len(ttes),
        fn_30s=fn_30s,
        fn_60s=fn_60s,
        clean_test_base_alarm_rows=0,
        instances=instances,
    )


def count_alarm_rows(output_rows: list[dict[str, Any]]) -> int:
    return sum(int(row["ids"]) for row in output_rows)


def run_view(
    *,
    config,
    view: str,
    max_formel_length: int,
    threshold_factor: float,
    cusum_factor: float,
    cpus: int,
    control_timestamp_shift_seconds: int,
    save_root: Path,
) -> EvaluationSummary:
    channels = pasad_scenario.default_channels_for_view(config, view)
    data_root = save_root / "data" / view
    output_root = save_root / "output" / view
    config_root = save_root / "config" / view

    train_rows = build_state_rows_for_split(
        config=config,
        view=view,
        split_name="training",
        channels=channels,
        control_timestamp_shift_seconds=control_timestamp_shift_seconds,
    )
    train_state_path = data_root / "training.state.gz"
    write_state_file(train_state_path, train_rows)

    scenario_key = config.scenario.lower()
    config_path = config_root / f"geco_{scenario_key}_{view}.json"
    config_payload = setup_geco_config(
        config_path=config_path,
        model_file_name=f"geco_{scenario_key}_{view}.model",
        max_formel_length=min(max_formel_length, len(channels) - 1),
        threshold_factor=threshold_factor,
        cusum_factor=cusum_factor,
        cpus=cpus,
    )
    geco = train_geco(
        config_path=config_path,
        config_payload=config_payload,
        train_state_path=train_state_path,
    )

    output_by_test: dict[str, list[dict[str, Any]]] = {}
    for split_idx in range(20):
        split_name = f"test_{split_idx:02d}"
        rows = build_state_rows_for_split(
            config=config,
            view=view,
            split_name=split_name,
            channels=channels,
            control_timestamp_shift_seconds=control_timestamp_shift_seconds,
        )
        input_path = data_root / f"{split_name}.state.gz"
        output_path = output_root / f"{split_name}.state.gz"
        write_state_file(input_path, rows)
        geco.cusum = {}
        geco.last_value = {}
        output_by_test[split_name] = run_live_on_state_file(
            geco=geco,
            input_path=input_path,
            output_path=output_path,
        )

    summary = evaluate_outputs(config=config, output_by_test=output_by_test)
    clean_rows = build_state_rows_for_split(
        config=config,
        view=view,
        split_name="test_base",
        channels=channels,
        control_timestamp_shift_seconds=control_timestamp_shift_seconds,
    )
    clean_input_path = data_root / "test_base.state.gz"
    clean_output_path = output_root / "test_base.state.gz"
    write_state_file(clean_input_path, clean_rows)
    geco.cusum = {}
    geco.last_value = {}
    clean_output_rows = run_live_on_state_file(
        geco=geco,
        input_path=clean_input_path,
        output_path=clean_output_path,
    )
    scenario_key = config.scenario.lower()
    summary.method_id = f"geco_{scenario_key}_{view}"
    summary.view = view
    summary.channels = list(channels)
    summary.max_formel_length = int(config_payload["GeCo"]["max_formel_length"])
    summary.threshold_factor = threshold_factor
    summary.cusum_factor = cusum_factor
    summary.cpus = cpus
    summary.control_timestamp_shift_seconds = control_timestamp_shift_seconds if view == "rec" else None
    summary.clean_test_base_alarm_rows = count_alarm_rows(clean_output_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GeCo on scenario-configured historian-only or historian+PVParser-recovered state.")
    parser.add_argument("--scenario", choices=tuple(pasad_scenario.SCENARIOS.keys()), required=True)
    parser.add_argument("--view", choices=("sup", "rec", "both"), default="both")
    parser.add_argument("--max-formel-length", type=int, default=3)
    parser.add_argument("--threshold-factor", type=float, default=1.4178738316276462)
    parser.add_argument("--cusum-factor", type=float, default=5.982371851051667)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--control-timestamp-shift-seconds", type=int, default=0)
    parser.add_argument("--save-root", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = pasad_scenario.get_config(args.scenario)
    save_root = args.save_root if args.save_root is not None else REPO_ROOT / "results" / f"geco_{args.scenario.lower()}"
    views = ("sup", "rec") if args.view == "both" else (args.view,)

    for view in views:
        summary = run_view(
            config=config,
            view=view,
            max_formel_length=args.max_formel_length,
            threshold_factor=args.threshold_factor,
            cusum_factor=args.cusum_factor,
            cpus=args.cpus,
            control_timestamp_shift_seconds=args.control_timestamp_shift_seconds,
            save_root=save_root,
        )
        out_path = save_root / "raw" / f"{summary.method_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True))
        print(json.dumps(asdict(summary), ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
