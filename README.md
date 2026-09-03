# CUDA-Accelerated Particle Swarm Optimization for EV Charging Station Placement

Optimal placement of electric vehicle (EV) charging stations across 50 real wards of Bengaluru, using a fully GPU-resident CUDA-accelerated Particle Swarm Optimization (PSO) framework — with a budget-matched Genetic Algorithm (GA) baseline and an archive-based Multi-Objective PSO (MOPSO) for cost-vs-coverage trade-off analysis.

> Charging-station placement is a combinatorial, NP-hard optimization problem. This project searches for near-optimal station layouts using swarm intelligence, and accelerates that search by keeping the entire PSO loop — not just fitness evaluation — resident on the GPU.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Methodology Details](#methodology-details)
- [Results in Detail](#results-in-detail)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

EV adoption has outpaced charging-infrastructure deployment, particularly in residential neighborhoods. This project models charging demand for 50 wards in Bengaluru — combining real ward-level population-density estimates with an hourly dual-peak temporal profile (morning commercial / evening residential charging) — and searches for the best set of `k` station locations to minimize a weighted objective of coverage loss, travel distance, and grid overload.

**Core contributions:**

- A **demand-driven placement model** using population-density estimates and an hourly dual-peak (Gaussian) temporal charging profile, with stations sized against the city-wide *coincident* peak hour (found to be 19:00), consistent with standard grid-planning practice.
- A **fully GPU-resident CUDA-PSO implementation** — particle positions, velocities, and personal bests stay on the GPU for the entire optimization loop (not just the fitness evaluation), using an on-device `xoroshiro128p` random-number generator so no per-iteration random draws cross the host–device boundary.
- A **fair, budget-matched Genetic Algorithm baseline** — identical solution encoding, fitness function, and total evaluation budget as PSO, for an apples-to-apples comparison.
- An **archive-based Multi-Objective PSO (MOPSO)** with adaptive-grid diversity preservation, producing a genuine Pareto front trading infrastructure cost against coverage loss.
- **Rigorous statistical validation** — paired *t*-tests across 10 repeated, identically-seeded runs to confirm results are real and not measurement noise.

---

## Key Results

| Metric | CPU-PSO | GPU-PSO | GA |
|---|---:|---:|---:|
| Mean fitness (lower = better) | 0.1371 | 0.1358 | 0.1326 |
| Mean time per run (s) | 0.859 | 0.247 | 1.147 |

- **Speedup:** 3.48× at the reference configuration (100 particles); scaling from **3.26× at 50 particles to 87.68× at 3200 particles**.
- **Solution quality:** CPU-PSO vs. GPU-PSO fitness difference is **not statistically significant** (paired *t*-test, *p* = 0.584) — the GPU implementation preserves optimization quality. The time difference **is** highly significant (*p* ≈ 6 × 10⁻⁶).
- **GA vs. PSO:** GA attains a modestly better mean fitness (*p* = 0.033 vs. CPU-PSO), attributed to PSO's lack of an explicit mutation operator on this discrete search space — not an implementation asymmetry. PSO remains the algorithm accelerated here because its uniform, per-particle update rule maps far more naturally onto GPU hardware than GA's sequential, data-dependent operators.
- **Demand-model ablation:** switching from a uniform-demand assumption to the full dual-peak temporal model reduces mean fitness by **69%**.
- **MOPSO Pareto front:** 13 non-dominated station layouts spanning cost 0.197–0.647 and coverage loss 0.099–0.365.

All numbers above are produced directly by the code — nothing is hand-entered.

---

## How It Works

1. **Represent a solution** as a `k`-dimensional vector of real numbers in `[0, 1]`. Each value is decoded to a ward index (`floor(value × n_wards)`); collisions are resolved by linear probing to guarantee exactly `k` distinct stations.
2. **Score a solution** with a single fitness value combining three demand-weighted terms:
   - **Coverage loss** — fraction of total demand *not* within 3 km (haversine distance) of a station.
   - **Distance** — demand-weighted mean distance to the nearest station, normalized by 15 km.
   - **Overload** — demand exceeding each station's 150 kW capacity, normalized across stations.
   ```
   fitness = 0.5 × coverage_loss + 0.3 × distance + 0.2 × overload
   ```
3. **Search for the best solution** using:
   - **PSO** — a swarm of particles converges via personal-best (`pbest`) and global-best (`gbest`) attraction, with inertia weight linearly annealed from 0.9 → 0.4.
   - **GA** — tournament selection, uniform crossover, single-gene mutation, and elitism, over the identical encoding and fitness function.
4. **Accelerate PSO on the GPU** by keeping the entire optimization loop — not just fitness evaluation — resident on the device across all iterations (see [Methodology Details](#methodology-details)).
5. **Trade off cost vs. coverage** with an archive-based MOPSO, allowing station count to vary (3–10) so cost and coverage form a genuine, non-trivial front.

---

## Architecture

```
                     ┌─────────────────────────┐
                     │   backend/core.py       │
                     │   single source of truth │
                     │  • 50-ward dataset       │
                     │  • demand model          │
                     │  • fitness function       │
                     │  • CPU PSO / GA / MOPSO   │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │    backend/gpu.py         │
                     │  CUDA kernels (Numba)     │
                     │  • fitness kernel         │
                     │  • velocity/position kernel│
                     │  • personal-best kernel    │
                     │  fully GPU-resident loop   │
                     └────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  Experiment driver        verify_consistency.py       Plotting module
  (paired CPU/GPU/GA runs,  (CPU vs GPU fitness         (all figures generated
   speedup sweeps, MOPSO,    cross-check)                 directly from result
   paired t-tests)                                         data)
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **Python** | Primary implementation language |
| **NumPy** | CPU-side numerical computation (PSO, GA, demand/fitness math) |
| **Numba (`numba.cuda`)** | JIT-compiles Python to CUDA GPU kernels; on-device RNG |
| **SciPy (`scipy.stats`)** | Paired *t*-tests for statistical significance |
| **Matplotlib** | All figures — demand curves, convergence, speedup, Pareto front, map, ablation |
| **Google Colab + NVIDIA GPU** | Execution environment for all CUDA experiments |

---

## Getting Started

### Prerequisites

- Python 3.9+
- An NVIDIA CUDA-capable GPU + CUDA toolkit (optional — the code detects GPU availability and falls back to CPU-only execution with a warning if none is found)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numba scipy numpy matplotlib
```

### Running

The project is structured as a Colab notebook (`Untitled55.ipynb`) that writes out a `backend/` package and then runs the full experiment pipeline. To reproduce:

```bash
# From within the notebook environment, or after extracting backend/:
python -m backend.verify_consistency   # sanity-check CPU vs GPU fitness agreement
# then run the experiment driver cell/script to reproduce all figures and tables
```

> **Note:** GPU-dependent results (speedup, GPU-PSO fitness) require an actual CUDA device. Running without one will print an explicit warning and silently fall back to CPU-only PSO rather than fabricating a GPU number.

---

## Project Structure

| File | Responsibility |
|---|---|
| `backend/core.py` | Dataset (50 wards), demand model, fitness function, CPU PSO, GA, MOPSO, particle decoding |
| `backend/gpu.py` | CUDA kernels and the fully GPU-resident PSO loop; GPU availability check + CPU fallback |
| Experiment driver | Runs CPU-PSO / GPU-PSO / GA for statistical comparison (10 paired runs), swarm-size and dataset-size sweeps, MOPSO run, paired *t*-tests |
| `backend/verify_consistency.py` | One-shot check confirming CPU and GPU fitness functions compute the same objective |
| Plotting module | Generates every figure directly from result data |

---

## Methodology Details

### Dataset
50 wards of Bengaluru, sampled at evenly spaced intervals across the full ward set to preserve geographic spread. Each ward has real latitude/longitude coordinates and a population-density estimate.

### Demand Model
- **Spatial:** demand scales with each ward's population-density estimate.
- **Temporal:** an hourly dual-peak model — a Gaussian centered at 08:00 (width 1.0 h, workplace/commercial charging) and a Gaussian centered at 19:00 (width 1.2 h, residential home charging). Each ward has a fixed, seeded commercial/residential land-use mix.
- **Design load:** the city-wide hourly demand curve is computed by summing all wards' hourly demand; the hour of maximum total demand (found to be 19:00, the *coincident peak*) is used as each ward's design-load snapshot — consistent with standard grid-planning practice of designing for coincident rather than per-ward peak demand.

### GPU-Resident CUDA Acceleration
Unlike designs that offload only fitness evaluation, this implementation keeps particle positions, velocities, personal-best positions, and personal-best fitness values resident on the GPU for the entire run:

- A **velocity/position update kernel** updates all particles in place using an on-device `xoroshiro128p` random-number generator — no per-iteration random draws cross the host–device boundary.
- A **fitness kernel** evaluates the objective directly on the device, replicating the exact CPU computation.
- Only the personal-best-fitness vector (one float per particle) and, when the global best improves, a single `k`-length row are copied back to the host each iteration. The full position/velocity matrices never leave the device mid-run.

This is the central reason the measured speedup scales meaningfully with swarm size instead of staying near 1×.

### Genetic Algorithm Baseline
Uses the identical decode function and fitness evaluation as PSO. Tournament selection (size 3), uniform crossover (rate 0.85), single-gene mutation (rate 0.10), 10% elitism. Population size and generation count are matched to PSO's swarm size and iteration count, giving both algorithms an identical total of 5,100 fitness evaluations.

### Multi-Objective PSO
An archive-based MOPSO following Coello Coello et al.'s adaptive-grid structure: an external non-dominated archive updated throughout optimization, Pareto-dominance-based personal-best updates, and crowding-based (grid-cell) tournament selection for leader choice, favoring less-crowded regions of the front. Station count is allowed to vary between 3 and 10 with a station-count-proportional infrastructure-cost term, so cost and coverage loss form a genuine trade-off.

### Statistical Validation
Paired *t*-tests (`scipy.stats.ttest_rel`) across 10 repeated runs, using identical random seeds per pair so CPU-PSO, GPU-PSO, and GA start from identical initial populations. This isolates the effect of the algorithm/hardware from run-to-run randomness.

---

## Results in Detail

### Convergence
CPU-PSO and GPU-PSO converge from an initial fitness of ≈0.203 to ≈0.137 / ≈0.136 respectively; GA reaches ≈0.133. Final solution quality is statistically indistinguishable between CPU-PSO and GPU-PSO.

### Speedup Scaling
| Particles | Speedup |
|---:|---:|
| 50 | 3.26× |
| 100 (reference, paired-run protocol) | 3.48× |
| 3200 | 87.68× |

A complementary sweep over dataset size (10–50 wards) yields speedups in the 4.6×–7.4× range.

### Ablation Study (demand model)
| Demand model | Mean fitness |
|---|---:|
| V1 — Uniform demand | 0.4253 ± 0.0094 |
| V2 — Density-based demand | 0.2704 ± 0.0083 |
| V3 — Dual-peak temporal demand | 0.1318 ± 0.0012 |

Temporal demand modeling reduces mean fitness by 51% relative to density-based demand alone, and by 69% relative to uniform demand.

### Pareto Front (MOPSO)
13 non-dominated station layouts, cost 0.197–0.647, coverage loss 0.099–0.365 — lower-cost (fewer-station) layouts show higher coverage loss, while higher station counts reduce coverage loss at greater infrastructure cost.

---

## Limitations & Future Work

- GA is currently CPU-only; a GPU-accelerated GA would test whether its fitness advantage persists once its own operators are parallelized.
- Evaluated on a 50-ward subset; scaling to the full ward set is a natural extension.
- Demand is modeled from population density and an assumed dual-peak temporal profile, not measured hourly charging telemetry — incorporating real telemetry where available would validate these assumptions.
- Station count is fixed (`k = 6`) for the single-objective PSO/GA comparison; joint optimization of station count and placement is left to the MOPSO formulation.

---

## Acknowledgments

Developed as part of academic work in the Department of Artificial Intelligence and Machine Learning. GPU experiments were run on Google Colab with NVIDIA GPU acceleration.
