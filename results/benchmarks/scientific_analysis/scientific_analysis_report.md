# Consolidated Scientific Analysis

## 1. Scope

This report consolidates the classical and geometric Grover benchmark campaigns. It uses the recorded JSON measurements without altering the source datasets.

The analysis covers three multi-scale scenarios:

| Scenario | Grid | Route length | Search space | Valid routes |
|---|---:|---:|---:|---:|
| G1_3x3_L4 | 3×3 | 4 | 256 | 3 |
| G2_4x4_L4 | 4×4 | 4 | 256 | 2 |
| G3_4x4_L6 | 4×4 | 6 | 4096 | 7 |

## 2. Experimental methodology

- A* and Dijkstra: 30 measured repetitions per scenario.
- Classical exhaustive enumeration: 30 repetitions per scenario.
- Classical warm-up runs are excluded from timing.
- Grover: five independent Aer CPU simulations for G1 and G2.
- G1: 1,024 shots per simulation, without an explicit MPS bond cap.
- G2: 32 shots per simulation, MPS bond dimension capped at 128.
- G3: no Grover transpilation or simulation.
- Classical timing intervals use the recorded Student-t confidence intervals.
- Grover success intervals use Wilson score confidence intervals.

## 3. Objective compatibility

A* and Dijkstra solve a shortest-path problem. Grover searches a fixed-length encoded route space. These are different objectives and their runtimes must not be interpreted as a direct speed ranking.

Classical exhaustive enumeration is the direct algorithmic comparator for Grover because both evaluate the same fixed-length route search space. The exhaustive baseline enumerates all encoded candidate routes and returns the valid indices.

This shared search space establishes objective compatibility, not direct execution-time equivalence. The classical baseline runs as CPU code, whereas Aer simulates a quantum circuit using a classical matrix-product-state method. Their measured wall-clock times therefore include fundamentally different computational workloads and must not be interpreted as a quantum speedup ratio.

The exhaustive timing covers the call to `GlobalClassicalOracle.find_valid_indices()` only. One warm-up call is excluded. Thirty measured calls are timed individually; result-consistency and route-validity checks occur after each timed call and are not included in the recorded enumeration time.

## 4. Classical results

| Scenario | Method | Mean | Median | 95% CI |
|---|---|---:|---:|---:|
| G1_3x3_L4 | astar | 17.81 µs | 17.30 µs | [15.63 µs, 19.99 µs] |
| G1_3x3_L4 | dijkstra | 16.60 µs | 15.91 µs | [15.46 µs, 17.74 µs] |
| G1_3x3_L4 | exhaustive | 1.741 ms | 503.57 µs | [-697.22 µs, 4.179 ms] |
| G2_4x4_L4 | astar | 18.59 µs | 19.04 µs | [16.84 µs, 20.35 µs] |
| G2_4x4_L4 | dijkstra | 19.02 µs | 19.41 µs | [16.93 µs, 21.12 µs] |
| G2_4x4_L4 | exhaustive | 487.14 µs | 477.00 µs | [466.59 µs, 507.69 µs] |
| G3_4x4_L6 | astar | 31.44 µs | 32.10 µs | [28.78 µs, 34.10 µs] |
| G3_4x4_L6 | dijkstra | 28.24 µs | 27.00 µs | [25.16 µs, 31.32 µs] |
| G3_4x4_L6 | exhaustive | 8.515 ms | 7.960 ms | [7.867 ms, 9.163 ms] |

### Interpretation

A* and Dijkstra remain extremely fast on these small grids. Their measured runtimes increase with scenario size, but the sample sizes and limited grid scales do not support extrapolation to large mazes.

G1 exhaustive enumeration has a pronounced high timing outlier. Its mean is approximately 1.741 ms, while its median is approximately 0.504 ms. The raw observation is retained; the median provides a more robust description of its central tendency. The conventional Student-t 95% confidence interval for the G1 mean extends below zero (approximately -0.697 to 4.179 ms). Negative execution time is physically impossible; this lower bound is a statistical artifact of applying a symmetric t interval to a highly skewed sample with an extreme observation. It is shown untruncated and should not be interpreted as a physically plausible time bound.

G2 exhaustive enumeration has a mean near 0.487 ms. G3, with 4,096 candidate sequences, has a mean near 8.515 ms. This illustrates the growth in exhaustive evaluation cost over the tested scenarios, not a universal complexity estimate.

## 5. Grover simulation results

| Scenario | Shots | Valid shots | Success | Wilson 95% CI | Mean simulation |
|---|---:|---:|---:|---:|---:|
| G1_3x3_L4 | 5120 | 5106 | 99.727% | [99.542%, 99.837%] | 6.794 s |
| G2_4x4_L4 | 160 | 160 | 100.000% | [97.655%, 100.000%] | 26.787 s |

G1 achieved 99.727% aggregate valid measurements over 5,120 shots. G2 achieved 100% observed valid measurements over 160 shots. This is an empirical result for the tested sample, not a guarantee of perfect success probability. Its Wilson interval is wider than G1's because the sample contains substantially fewer shots. Furthermore, G2 uses an MPS bond-dimension cap of 128, so its simulation is approximate and should be interpreted diagnostically.

The five simulator runs use 32 shots each. The aggregate Wilson interval describes the pooled shot outcomes; it does not quantify all sources of simulator approximation or variability between independent runs.

These observations describe the tested simulator configurations. They do not establish performance on quantum hardware.
## 6. Circuit structure

| Scenario | Qubits | Initial depth | Transpiled depth | Transpiled operations |
|---|---:|---:|---:|---:|
| G1_3x3_L4 | 18 | 955 | 24160 | 26958 |
| G2_4x4_L4 | 20 | 995 | 24768 | 30982 |
| G3_4x4_L6 | 26 | 3243 | Not performed | Not performed |

The circuit grows from 18 qubits and initial depth 955 in G1 to 26 qubits and initial depth 3,243 in G3. G3 was not transpiled or simulated because of the computational resource constraints of the experimental environment.

## 7. Important implementation limitation

The Grover circuit construction process performs classical solution counting to determine the number of Grover iterations. Consequently, the measured pipeline is not a fully end-to-end quantum search procedure. This classical preprocessing must be considered when interpreting algorithmic performance.

## 8. Threats to validity and limitations

1. Small grids and search spaces limit generalization.
2. A* and Dijkstra do not solve the same objective as fixed-length Grover.
3. Aer MPS simulation is a classical simulation of a quantum circuit.
4. G2 uses an approximate MPS configuration and only 160 total shots.
5. G3 has no transpilation or simulation measurements.
6. G1 exhaustive timing contains a strong outlier.
7. CPU timing depends on the host machine, Python runtime and system load.
8. Grover iteration selection uses classically counted solutions.
9. No hardware execution or quantum advantage has been demonstrated.

## 9. Conclusion

The experiments demonstrate a working geometric Grover circuit for the tested small fixed-length route-search instances, with high observed valid-measurement proportions in G1 and G2. They also show that classical planners are extremely efficient on these small mazes and that classical exhaustive enumeration remains inexpensive at the tested search-space sizes.

The present results support a methodological and implementation comparison, not a claim of quantum computational advantage. Larger instances, fair end-to-end baselines, hardware-aware resource accounting and additional independent experiments would be required for stronger conclusions.

## 10. Experimental computing environment

- Host processor: Intel Core i5-1135G7, 2.40 GHz, 4 cores and 8 threads.
- Execution environment: Ubuntu under WSL2, x86_64.
- Python: 3.12.3.
- Qiskit: 2.6.2; Qiskit Aer: 0.18.4.
- NumPy: 2.5.5; SciPy: 1.18.1; Matplotlib: 3.11.2.
- Aer simulation method: `matrix_product_state`.
- Aer MPS OpenMP threads: 1; configured memory limit: 7,000 MB.

Reported execution times are environment-dependent. They should be reproduced on the same or a documented equivalent environment before making timing comparisons across machines.

## 11. Reproducibility

Source datasets:

- `results/benchmarks/classical_vs_quantum/benchmark_classical_vs_quantum_statistics.json`
- `results/benchmarks/multi_scale_statistics/g1_3x3_l4_statistics.json`
- `results/benchmarks/multi_scale_statistics/g2_4x4_l4_statistics.json`
- `results/benchmarks/multi_scale_statistics/g3_4x4_l6_statistics.json`

Figures and CSV tables are generated by `experiments/benchmark_scientific_analysis/analyze_results.py`.
