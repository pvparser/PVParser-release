# Attack Detection Artifacts

This directory contains the attack-generation, attack-detection, and enhancement code used for the attack-detection experiments in the paper.

## Attack Generation

The `s2`-`s5` directories correspond to the paper's attack-generation scenarios:

| Directory | Attack code | Contents |
| --- | --- | --- |
| `s2/` | Attack 2 generation | Supervisory historian, supervisory traffic, and control-layer attack-generation code. |
| `s3/` | Attack 4 generation | Supervisory historian, supervisory traffic, and control-layer attack-generation code. |
| `s4/` | Attack 2 generation | Supervisory historian, supervisory traffic, and control-layer attack-generation code. |
| `s5/` | Attack 4 generation | Supervisory historian, supervisory traffic, and control-layer attack-generation code. |

Each scenario directory is split by data source:

- `supervisory_historian_attack/`: generates manipulated historian CSV traces.
- `supervisory_traffic_attack/`: injects the corresponding changes into supervisory traffic.
- `control_attack/`: injects the corresponding changes into control-layer traffic.

## Detection and Enhancement

The remaining method directories contain attack-detection baselines, reproduced evaluation scripts, and PVParser-enhanced variants:

| Directory | Purpose |
| --- | --- |
| `method_clc/` | CLC-based attack-detection code. |
| `method_passad/` | PASAD-based attack-detection and PVParser-enhanced evaluation code. |
| `method_geco/` | GeCo-based attack-detection and PVParser-enhanced evaluation code. |
| `method_nnd/` | NND/RAID 2024 reproduction and evaluation code. |
| `time_alignment/` | Shared SWaT historian-to-traffic time-alignment utilities used by the generation and evaluation scripts. |

