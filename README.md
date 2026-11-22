# Secure Dynamic Multi-Agent Maze Race

A research-oriented framework for multi-agent navigation, dynamic obstacle avoidance, coordination, and quantum-classical planning in maze environments.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

---

## 1. Overview

Secure Dynamic Multi-Agent Maze Race is a Python research project exploring navigation and coordination strategies for multiple autonomous agents operating in maze environments.

The framework combines:

- Classical path-planning algorithms.
- Quantum and hybrid quantum-classical navigation approaches.
- Dynamic environment modelling.
- Moving-obstacle handling.
- Multi-agent coordination and traffic management.
- Collision detection and safety mechanisms.
- Benchmarking, performance metrics, and visualization.

The project aims to provide a controlled experimental environment for studying autonomous multi-agent navigation.

## 2. Research Objectives

The main research objectives are:

1. Compare classical and quantum-classical navigation strategies.
2. Investigate coordination between multiple autonomous agents.
3. Study navigation in environments with changing walls.
4. Handle moving obstacles in shared corridors.
5. Reduce collisions, conflicting movements, and deadlocks.
6. Evaluate navigation performance using reproducible experiments.

## 3. Target Scenario

The final target benchmark is designed around the following specifications:

| Parameter | Target |
|---|---|
| Maze dimensions | 50 × 50 cells |
| Number of robots | 10 |
| Entrance gates | 10 distinct entrances |
| Robot assignment | One robot per entrance |
| Destination | One shared objective at the maze center |
| Dynamic walls | Update every 2 seconds |
| Moving obstacles | Present on selected corridors |
| Navigation objective | Minimize travel cost |
| Safety objective | Avoid collisions and unsafe movements |
| Coordination | Shared-goal multi-agent navigation |

The environment must preserve feasible routes toward the common destination while distinguishing static walls, dynamic walls, and moving obstacles.

Implementation status: The target scenario is under development. Its specifications must not be confused with the historical experiments currently present in the repository.

## 4. Existing Experimental Baselines

### 4.1 Ten-Agent Static Benchmark

The historical ten-agent benchmark uses:

- A 25 × 25 maze.
- Ten agents with individually assigned destinations.
- Five classical planners.
- Five hybrid quantum-classical planners.
- Random seed: 42.
- Maximum simulation cycles: 2,000.
- Simulation time step: 1/30 second.

This benchmark does not include the final shared central destination or dynamic obstacles.

### 4.2 Preliminary Dynamic Experiment

A separate preliminary experiment uses:

- A 7 × 7 maze.
- Two agents with opposing destinations.
- One moving obstacle.
- An obstacle update interval of 0.10 second.
- Maximum simulation cycles: 600.

This experiment is a preliminary dynamic pilot, not the final ten-agent benchmark.

## 5. Navigation Algorithms

### 5.1 Classical Algorithms

| Agent | Algorithm |
|---|---|
| R01 | A* |
| R02 | Dijkstra |
| R03 | RRT* |
| R04 | D* Lite |
| R05 | Bidirectional BFS |

### 5.2 Hybrid Quantum-Classical Algorithms

| Agent | Algorithm |
|---|---|
| R06 | Grover-assisted Navigation |
| R07 | Global Quantum Walk |
| R08 | Hierarchical Local Quantum Walk |
| R09 | QAOA Navigation |
| R10 | Variational Quantum Classifier Navigation |

Quantum methodology note: Quantum circuits may be simulated on classical hardware. Hybrid quantum-classical methods must not be presented as purely quantum algorithms. This repository does not claim quantum hardware execution or quantum speedup.

## 6. Project Architecture

```text
SECURE_DYNAMIC_MAZE/
├── agents/
├── config/
├── docs/
├── environment/
├── experiments/
├── metrics/
├── planners/
│   ├── classical/
│   └── quantum/
├── results/
├── scripts/
├── security/
├── simulation/
├── tests/
├── visualization/
├── requirements-base.txt
├── requirements-quantum.txt
├── requirements-optimization.txt
└── README.md
```

### Main Components

- agents/: autonomous agent implementations.
- config/: general simulation configuration.
- environment/: maze and dynamic environment models.
- experiments/: benchmark scenarios and experiments.
- metrics/: performance measurement.
- planners/classical/: classical navigation algorithms.
- planners/quantum/: quantum and hybrid navigation algorithms.
- results/: experimental outputs.
- security/: safety-related components.
- simulation/: simulation and multi-agent coordination.
- tests/: automated tests.
- visualization/: visualization components.

Experiment-specific parameters may override general settings in config/settings.py.

## 7. Technology Stack

The project uses Python and scientific-computing libraries, including:

- NumPy and SciPy.
- NetworkX.
- Matplotlib.
- Pygame.
- Qiskit and Qiskit Aer.
- PennyLane.
- CasADi.
- OSQP.
- Pytest.

The exact dependency versions are specified in the requirements files.

## 8. Installation

### 8.1 Clone the Repository

```bash
git clone https://github.com/khourdouch-amine/Secure-Dynamic-Multi-Agent-Maze-Race.git
cd Secure-Dynamic-Multi-Agent-Maze-Race
```

### 8.2 Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 8.3 Install Base Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
```

### 8.4 Additional Dependencies

The repository also contains:

- requirements-quantum.txt
- requirements-optimization.txt

These files provide pinned dependency sets for experiments requiring quantum-computing or optimization libraries.

Review the dependency profiles before installation. They should not automatically be treated as small additive packages.

## 9. Running Experiments

Run the following commands from the project root with the virtual environment activated.

### Ten-Agent Static Benchmark

```bash
python experiments/benchmark_secure_multi_agent/benchmark_10_agents.py
```

### Preliminary Dynamic Multi-Agent Experiment

```bash
python experiments/benchmark_secure_multi_agent/benchmark_dynamic_multi_agent.py
```

### Classical Versus Quantum Benchmark

```bash
python experiments/benchmark_classical_vs_quantum/benchmark_classical_vs_quantum.py
```

### Global Grover Benchmark

```bash
python experiments/grover_global_benchmark.py
```

Results are saved in the output directories configured by each experiment.

## 10. Testing

Run the automated tests from the project root:

```bash
pytest -q
```

Test results depend on the code version, installed dependencies, and execution environment.

## 11. Results and Reproducibility

The repository contains historical experimental outputs in JSON and CSV formats, including:

- Ten-agent static benchmark results.
- Preliminary dynamic multi-agent results.
- Classical-versus-quantum comparisons.
- Multi-scale quantum experiments.
- Scientific analysis reports.
- Experimental checkpoints.

Some experiments use fixed random seeds to support reproducibility.

Comparisons must account for differences in maze dimensions, agent configurations, simulation conditions, and computational budgets.

## 12. Development Roadmap

The planned development stages are:

1. Consolidate existing maze, agent, and planner interfaces.
2. Implement the 50 × 50 maze.
3. Create ten distinct entrance gates.
4. Assign one robot to each entrance.
5. Implement the shared central destination.
6. Implement dynamic-wall updates every two seconds.
7. Integrate moving obstacles on selected corridors.
8. Improve multi-agent coordination and collision avoidance.
9. Implement and evaluate deadlock recovery.
10. Conduct reproducible benchmark experiments.
11. Produce validated comparative reports and visualizations.

## 13. Limitations

- The historical 25 × 25 benchmark does not represent the final target scenario.
- The preliminary 7 × 7 dynamic experiment is not a validation of the ten-agent scenario.
- Quantum circuit simulation on classical hardware is not equivalent to execution on a quantum processor.
- Performance comparisons require consistent experimental conditions.
- The final target benchmark remains under development.

## 14. License

No license has currently been specified for this repository. Reuse, redistribution, and modification permissions have therefore not been explicitly granted.

## 15. Author

Amine Khourdouch

GitHub: https://github.com/khourdouch-amine

Project repository:

https://github.com/khourdouch-amine/Secure-Dynamic-Multi-Agent-Maze-Race
