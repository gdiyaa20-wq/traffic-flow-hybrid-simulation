# Hybrid Traffic Flow Simulation (LWR + IDM)

A hybrid macroscopic-microscopic traffic simulation that models urban traffic flow by combining a continuum (fluid-based) model with an agent-based (per-vehicle) model, validated against real-world trajectory data.

## Overview

Most traffic models pick one of two lenses: **macroscopic** models treat traffic as a continuous fluid (fast, scalable, but no individual vehicle detail), while **microscopic** models simulate each vehicle individually (realistic, but computationally expensive at scale). This project builds a **hybrid framework** that uses both — a macroscopic model for the open road and a microscopic model near intersections/signals — with a transition zone that converts between the two representations without breaking vehicle count or momentum continuity.

## Key Components

- **Macroscopic layer:** Lighthill-Whitham-Richards (LWR) model, a conservation-law PDE solved numerically using the Godunov scheme, to model traffic density and flow.
- **Microscopic layer:** Intelligent Driver Model (IDM) to simulate individual vehicle acceleration, braking, and car-following behaviour, extended with multi-lane logic and signalized intersections.
- **Transition mechanism:** Converts macroscopic density into discrete vehicles (and back) while conserving vehicle count and momentum, so there's no discontinuity at the boundary.
- **Real-time data integration:** Vehicle trajectories from the NGSIM dataset are smoothed with a Gaussian kernel and overlaid on the simulation to validate and correct it in real time.
- **Validation:** Simulation accuracy measured against real trajectory data using RMSE and MAPE (achieved MAPE below 8% across all checkpoints tested).

## How to Run

```bash
pip install numpy matplotlib scipy
```

```python
from hybrid_traffic_simulation import TrafficSimulation

sim = TrafficSimulation(road_length=500, num_lanes=2, dt=0.5)
sim.animate(frames=200)  # run inside Jupyter/Colab to render the animation
```

## Tech Stack

- **Python**
- **NumPy** — vectorized state updates (position, velocity)
- **Matplotlib** — real-time plotting and animation of density profiles and vehicle trajectories
- **SciPy** — Gaussian smoothing and interpolation for real-time data correction

## What's in this repo

- `hybrid_traffic_simulation.py` — the complete simulation: LWR macroscopic model, IDM microscopic model, the macro-micro transition logic, multi-lane support, and signalized intersections, all in one class (`TrafficSimulation`).
- `Traffic_Flow_Modelling.pdf` — full dissertation write-up: methodology, pseudocode, results, and discussion.

## Results (Summary)

| Checkpoint | RMSE (veh/km) | MAPE (%) |
|---|---|---|
| Point A | 3.1 | 6.4 |
| Point B | 2.8 | 5.9 |
| Point C | 3.6 | 7.2 |

## Background

This was originally completed as a Semester VI capstone dissertation for a B.Sc. in Applied Mathematical Computing at NMIMS, Mumbai (April 2025), under the guidance of Dr. Debasmita Mukherjee.

## References

See `Traffic_Flow_Modelling.pdf` for the full literature review and citations (Laval & Daganzo 2006; Diaz et al. 2014; Bekiaris-Liberis et al. 2018; Treiber et al. 2000; Seo et al. 2018; Punzo & Ciuffo 2011).
