# PVParser-Release

This repository contains the artifact release for the manuscript *Recovering Process Variables from Industrial Network Traffic via Search-Based Optimization*. It is intended to support artifact inspection during peer review and is aligned with the Open Science statement in the paper.

## Artifacts Provided

The repository includes:

- the PVParser implementation for period identification, payload extraction, field inference, payload-to-process-variable alignment, and evaluation;
- attack-generation, attack-detection, and PVParser-enhanced detection code under `src/attack_detection/`;
- the Python environment specification in `environment.yml`;
- recovered outputs produced by PVParser under `src/data/`;
- reviewer-facing dataset notes under `dataset/`.

The recovered outputs in `src/data/` are included so that reviewers can inspect intermediate and final experimental artifacts without needing to rerun every pipeline stage from raw traffic.

## Repository Layout

| Path | Description |
| --- | --- |
| `src/period_identification/` | Periodic communication pattern identification. |
| `src/protocol_field_inference/` | Payload extraction, payload combination, and field inference. |
| `src/payload_inference/` | Merging, alignment, and process-variable field recovery. |
| `src/performance_evaluation/` | Evaluation scripts used to compute reported metrics. |
| `src/attack_detection/` | Attack-generation, attack-detection, and PVParser-enhanced detection artifacts. |
| `src/data/` | Recovered outputs and experiment artifacts produced by PVParser. |
| `dataset/` | Dataset notes and redistributable SCADA files. |

## Data Availability

The paper evaluates PVParser on SCADA Network, SWaT, and WADI industrial CPS datasets.

- SCADA Network files included under `dataset/scada/` are provided as reviewer-facing artifacts.
- SWaT and WADI are not redistributed in this repository. They originate from external research testbeds and must be requested from the official providers subject to their access procedures and usage conditions.
- See `dataset/README.md` for the SWaT and WADI download notes.

## Evaluation Scope

Because the full SWaT and WADI datasets cannot be redistributed, this artifact focuses on the implementation, environment specification, recovered outputs, and workflow needed to inspect the core methodology and reported experimental results. Researchers who obtain legitimate access to the original datasets can use the released code and artifacts to further verify the complete workflows.

This repository should remain fixed during the double-blind review period.
