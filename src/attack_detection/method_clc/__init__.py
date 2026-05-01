"""Cross-Level Consistency Checker (CLC) for attack_detection."""

from .clc import (
    CLCConfig,
    CLCModel,
    PairCalibration,
    PairSpec,
    apply_alarm_logic,
    evaluate_attack_intervals,
    extract_alarm_events,
    fit_clc_model,
    load_config,
    load_pair_specs,
    load_timeseries_csv,
    prepare_segment_from_dataframes,
    run_clc_pipeline,
    save_clc_artifacts,
    score_segment,
)

__all__ = [
    "CLCConfig",
    "CLCModel",
    "PairCalibration",
    "PairSpec",
    "apply_alarm_logic",
    "evaluate_attack_intervals",
    "extract_alarm_events",
    "fit_clc_model",
    "load_config",
    "load_pair_specs",
    "load_timeseries_csv",
    "prepare_segment_from_dataframes",
    "run_clc_pipeline",
    "save_clc_artifacts",
    "score_segment",
]
