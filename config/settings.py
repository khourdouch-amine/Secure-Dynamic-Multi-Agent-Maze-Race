"""
Central configuration for Secure Dynamic Multi-Agent Maze Race.
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"


# ============================================================
# SIMULATION SETTINGS
# ============================================================

SIMULATION_DURATION = 30.0
TARGET_FPS = 30
RANDOM_SEED = 42


# ============================================================
# MAZE SETTINGS
# ============================================================

MAZE_WIDTH = 25
MAZE_HEIGHT = 25
CELL_SIZE = 24

DYNAMIC_WALL_INTERVAL = 5.0


# ============================================================
# MULTI-AGENT SETTINGS
# ============================================================

NUMBER_OF_AGENTS = 10
SAFETY_DISTANCE = 1.0


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

WINDOW_WIDTH = MAZE_WIDTH * CELL_SIZE
WINDOW_HEIGHT = MAZE_HEIGHT * CELL_SIZE

WINDOW_TITLE = "Secure Dynamic Multi-Agent Maze Race"


# ============================================================
# DYNAMIC ENVIRONMENT SETTINGS
# ============================================================

NUMBER_OF_DYNAMIC_WALLS = 15
NUMBER_OF_MOVING_OBSTACLES = 12

OBSTACLE_UPDATE_INTERVAL = 0.35


# ============================================================
# VALIDATION
# ============================================================

assert SIMULATION_DURATION > 0
assert TARGET_FPS > 0
assert MAZE_WIDTH >= 5
assert MAZE_HEIGHT >= 5
assert CELL_SIZE > 0
assert DYNAMIC_WALL_INTERVAL > 0
assert NUMBER_OF_AGENTS >= 1
assert SAFETY_DISTANCE > 0
assert NUMBER_OF_DYNAMIC_WALLS >= 1
assert NUMBER_OF_MOVING_OBSTACLES >= 1
assert OBSTACLE_UPDATE_INTERVAL > 0
