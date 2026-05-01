#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pasad_scenario


@dataclass
class SearchCandidate:
    scenario: str
    description: str
    channels: list[str]
    vote: int
    tp: int
    fp: int
    tn: int
    fn: int
    acc: float
    f1: float
    tte_mean: float | None
    strict_tp: int
    strict_aer: float
    blind_clean_any_alarm: bool
    blind_clean_alarm_seconds: int


def load_channel_config(config: Path, scenario: str) -> dict[str, dict[str, int | None]]:
    payload = json.loads(config.read_text())
    view = payload.get("view")
    if view != "rec":
        raise ValueError(f"Only rec view is supported, got {view!r}")

    scenario_cfg = pasad_scenario.get_config(scenario)
    defaults = payload.get("defaults", {})
    overrides = payload.get("channels", {})
    resolved: dict[str, dict[str, int | None]] = {}

    for channel in scenario_cfg.rec_channels:
        raw = dict(defaults)
        raw.update(overrides.get(channel, {}))
        lag = raw.get("lag")
        rank = raw.get("rank")
        train_length = raw.get("train_length")
        if lag is None or rank is None:
            raise ValueError(f"Missing lag/rank for channel {channel} in {config}")
        resolved[channel] = {
            "lag": int(lag),
            "rank": int(rank),
            "train_length": None if train_length is None else int(train_length),
        }
    return resolved


def train_channel_models(
    *,
    config: pasad_scenario.ScenarioConfig,
    channel_config: dict[str, dict[str, int | None]],
    epsilon: float,
    control_timestamp_shift_seconds: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    attack_log = pasad_scenario.load_attack_log(config)
    cache: dict[str, dict[str, Any]] = {}

    for channel, params in channel_config.items():
        train_trace = pasad_scenario.load_channel_trace(
            config=config,
            view="rec",
            split_name="training",
            channel=channel,
            control_timestamp_shift_seconds=control_timestamp_shift_seconds,
        )
        model_train_df, calibration_df = pasad_scenario.split_training_and_calibration(
            train_df=train_trace,
            lag=int(params["lag"]),
            threshold_source="train_holdout",
            train_length=params["train_length"],
        )
        model_summary, model_cache = pasad_scenario.train_pasada_model(
            series=model_train_df["value"].to_numpy(dtype=float),
            channel=channel,
            lag=int(params["lag"]),
            rank=int(params["rank"]),
            calibration_series=calibration_df["value"].to_numpy(dtype=float),
            epsilon=epsilon,
        )
        cache[channel] = {
            **model_cache,
            "summary": asdict(model_summary),
        }
    return cache, attack_log


def collect_channel_hits(
    *,
    config: pasad_scenario.ScenarioConfig,
    channel_config: dict[str, dict[str, int | None]],
    model_cache: dict[str, dict[str, Any]],
    attack_log: pd.DataFrame,
    control_timestamp_shift_seconds: int,
) -> tuple[dict[str, set[int]], dict[str, dict[str, dict[str, float | None]]]]:
    clean_alarm_seconds: dict[str, set[int]] = {}
    instance_hits: dict[str, dict[str, dict[str, float | None]]] = {}

    for channel in channel_config:
        clean_trace = pasad_scenario.load_channel_trace(
            config=config,
            view="rec",
            split_name="test_base",
            channel=channel,
            control_timestamp_shift_seconds=control_timestamp_shift_seconds,
        )
        cached = model_cache[channel]
        clean_scores = pasad_scenario.score_pasada_series(
            basis=cached["basis"],
            centroid_projection=cached["centroid_projection"],
            weights=cached["weights"],
            train_tail=cached["train_tail"],
            eval_series=clean_trace["value"].to_numpy(dtype=float),
        )
        indices = np.flatnonzero(clean_scores >= cached["threshold"])
        clean_alarm_seconds[channel] = set(np.floor(clean_trace.iloc[indices]["timestamp_epoch"]).astype(int).tolist())

    for _, row in attack_log.iterrows():
        test_name = Path(str(row["output_file"])).stem
        attack_start_epoch, attack_end_epoch = pasad_scenario.attack_window_epoch(row)
        per_channel: dict[str, dict[str, float | None]] = {}
        for channel in channel_config:
            cached = model_cache[channel]
            trace = pasad_scenario.load_channel_trace(
                config=config,
                view="rec",
                split_name=test_name,
                channel=channel,
                control_timestamp_shift_seconds=control_timestamp_shift_seconds,
            )
            pre_epoch, in_epoch = pasad_scenario.first_alarm_epochs_streaming(
                basis=cached["basis"],
                centroid_projection=cached["centroid_projection"],
                weights=cached["weights"],
                train_tail=cached["train_tail"],
                threshold=float(cached["threshold"]),
                trace_df=trace,
                attack_start_epoch=attack_start_epoch,
                attack_end_epoch=attack_end_epoch,
            )
            per_channel[channel] = {
                "pre": pre_epoch,
                "attack": in_epoch,
            }
        instance_hits[test_name] = per_channel

    return clean_alarm_seconds, instance_hits


def aggregate_candidate(
    *,
    config: pasad_scenario.ScenarioConfig,
    attack_log: pd.DataFrame,
    instance_hits: dict[str, dict[str, dict[str, float | None]]],
    clean_alarm_seconds: dict[str, set[int]],
    channels: tuple[str, ...],
    vote: int,
) -> SearchCandidate:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    strict_tp = 0
    ttes: list[float] = []

    for _, row in attack_log.iterrows():
        test_name = Path(str(row["output_file"])).stem
        attack_start_epoch, _ = pasad_scenario.attack_window_epoch(row)
        hits = instance_hits[test_name]
        pre_epochs = sorted(epoch for channel in channels if (epoch := hits[channel]["pre"]) is not None)
        attack_epochs = sorted(epoch for channel in channels if (epoch := hits[channel]["attack"]) is not None)

        pre_detected = len(pre_epochs) >= vote
        attack_detected = len(attack_epochs) >= vote

        if pre_detected:
            fp += 1
        else:
            tn += 1

        if attack_detected:
            tp += 1
            kth_attack_epoch = float(attack_epochs[vote - 1])
            ttes.append(kth_attack_epoch - attack_start_epoch)
            if not pre_detected:
                strict_tp += 1
        else:
            fn += 1

    second_counts: Counter[int] = Counter()
    for channel in channels:
        for second in clean_alarm_seconds[channel]:
            second_counts[second] += 1
    clean_alarm_count = sum(1 for count in second_counts.values() if count >= vote)

    total = tp + fp + tn + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return SearchCandidate(
        scenario=config.scenario,
        description=config.description,
        channels=list(channels),
        vote=vote,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        acc=(tp + tn) / total,
        f1=f1,
        tte_mean=None if not ttes else float(np.mean(ttes)),
        strict_tp=strict_tp,
        strict_aer=strict_tp / len(attack_log),
        blind_clean_any_alarm=clean_alarm_count > 0,
        blind_clean_alarm_seconds=clean_alarm_count,
    )


def search_candidates(
    *,
    config: pasad_scenario.ScenarioConfig,
    attack_log: pd.DataFrame,
    instance_hits: dict[str, dict[str, dict[str, float | None]]],
    clean_alarm_seconds: dict[str, set[int]],
    require_ctrl: bool,
    require_sup: bool,
    min_size: int,
    max_size: int,
) -> list[SearchCandidate]:
    all_channels = config.rec_channels
    out: list[SearchCandidate] = []
    for size in range(min_size, max_size + 1):
        for subset in itertools.combinations(all_channels, size):
            has_ctrl = any(channel in config.control_rec_channels for channel in subset)
            has_sup = any(channel in config.sup_channels for channel in subset)
            if require_ctrl and not has_ctrl:
                continue
            if require_sup and not has_sup:
                continue
            for vote in range(1, len(subset) + 1):
                out.append(
                    aggregate_candidate(
                        config=config,
                        attack_log=attack_log,
                        instance_hits=instance_hits,
                        clean_alarm_seconds=clean_alarm_seconds,
                        channels=subset,
                        vote=vote,
                    )
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Search PASAD rec subsets and vote thresholds from per-channel first-hit times.")
    parser.add_argument("--scenario", choices=tuple(pasad_scenario.SCENARIOS.keys()), required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--control-timestamp-shift-seconds", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=7)
    parser.add_argument("--require-ctrl", action="store_true", default=True)
    parser.add_argument("--require-sup", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    config = pasad_scenario.get_config(args.scenario)
    channel_config = load_channel_config(args.config_json, args.scenario)
    if args.max_size > len(config.rec_channels):
        args.max_size = len(config.rec_channels)

    model_cache, attack_log = train_channel_models(
        config=config,
        channel_config=channel_config,
        epsilon=args.epsilon,
        control_timestamp_shift_seconds=args.control_timestamp_shift_seconds,
    )
    clean_alarm_seconds, instance_hits = collect_channel_hits(
        config=config,
        channel_config=channel_config,
        model_cache=model_cache,
        attack_log=attack_log,
        control_timestamp_shift_seconds=args.control_timestamp_shift_seconds,
    )
    candidates = search_candidates(
        config=config,
        attack_log=attack_log,
        instance_hits=instance_hits,
        clean_alarm_seconds=clean_alarm_seconds,
        require_ctrl=args.require_ctrl,
        require_sup=args.require_sup,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    candidates.sort(
        key=lambda item: (
            item.acc,
            item.f1,
            item.strict_aer,
            -item.blind_clean_alarm_seconds,
            -(999999.0 if item.tte_mean is None else item.tte_mean),
        ),
        reverse=True,
    )

    payload = {
        "scenario": config.scenario,
        "description": config.description,
        "config_json": str(args.config_json),
        "control_timestamp_shift_seconds": args.control_timestamp_shift_seconds,
        "channel_config": channel_config,
        "top": [asdict(candidate) for candidate in candidates[: args.top_k]],
    }
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
