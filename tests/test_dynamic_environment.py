from environment.maze import Maze, FREE, MOVING_OBSTACLE
from environment.dynamic_elements import DynamicEnvironment


def test_wall_update_does_not_reset_obstacle_clock():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze)

    env.update(4.9)
    result = env.update(5.25)

    assert result["walls_updated"] is True
    assert result["obstacles_updated"] is True
    assert env.last_obstacle_update == 5.25
    assert len(env.wall_positions) == 15
    assert len(env.obstacle_positions) == 12


def test_dynamic_environment_can_be_initialized_from_agents():
    from types import SimpleNamespace

    maze = Maze(width=25, height=25, seed=42)

    free_cells = [
        (x, y)
        for y in range(maze.height)
        for x in range(maze.width)
        if maze.is_walkable(x, y)
    ]

    starts = free_cells[2:4]
    goals = free_cells[-4:-2]

    agents = [
        SimpleNamespace(start=starts[0], goal=goals[0]),
        SimpleNamespace(start=starts[1], goal=goals[1]),
    ]

    env = DynamicEnvironment.from_agents(
        maze,
        agents,
        seed=42,
    )

    expected_protected = set(starts) | set(goals)

    assert expected_protected.issubset(env.protected_cells)
    assert set(zip(starts, goals)).issubset(
        set(env.connectivity_pairs)
    )

    for x, y in expected_protected:
        assert maze.is_walkable(x, y)

    assert len(env.wall_positions) == 15
    assert len(env.obstacle_positions) == 12


def test_dynamic_environment_protects_current_agent_positions():
    from types import SimpleNamespace

    maze = Maze(width=25, height=25, seed=42)

    agent = SimpleNamespace(
        agent_id="R1",
        start=(2, 2),
        position=(5, 5),
        goal=(10, 10),
    )

    env = DynamicEnvironment.from_agents(
        maze,
        [agent],
        seed=42,
    )

    assert (2, 2) in env.protected_cells
    assert (5, 5) in env.protected_cells
    assert (10, 10) in env.protected_cells

    assert ((5, 5), (10, 10)) in env.connectivity_pairs

    assert maze.is_walkable(2, 2)
    assert maze.is_walkable(5, 5)
    assert maze.is_walkable(10, 10)

    assert len(env.wall_positions) == 15
    assert len(env.obstacle_positions) == 12


def test_obstacle_registry_preserves_ids_after_movement():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    initial_registry = dict(env.obstacle_registry)
    expected_ids = {f"O{index:02d}" for index in range(12)}

    assert set(initial_registry) == expected_ids
    assert len(set(initial_registry.values())) == 12
    assert set(initial_registry.values()) == env.obstacle_positions

    env.update_obstacles()

    assert set(env.obstacle_registry) == expected_ids
    assert len(env.obstacle_registry) == 12
    assert len(set(env.obstacle_registry.values())) == 12
    assert set(env.obstacle_registry.values()) == env.obstacle_positions

    for obstacle_id, old_position in initial_registry.items():
        new_position = env.obstacle_registry[obstacle_id]

        distance = (
            abs(new_position[0] - old_position[0])
            + abs(new_position[1] - old_position[1])
        )

        assert distance in (0, 1)


def test_obstacle_registry_preserves_ids_after_wall_regeneration():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    initial_registry = dict(env.obstacle_registry)

    env.update_walls()

    assert set(env.obstacle_registry) == set(initial_registry)
    assert len(env.obstacle_registry) == 12
    assert len(set(env.obstacle_registry.values())) == 12
    assert set(env.obstacle_registry.values()) == env.obstacle_positions
    assert len(env.wall_positions) == 15


def test_obstacle_keeps_identity_when_previous_cell_is_occupied():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    obstacle_id = "O00"
    old_position = env.obstacle_registry[obstacle_id]
    occupied = {old_position}

    env.update_obstacles(occupied_cells=occupied)

    assert set(env.obstacle_registry) == {
        f"O{index:02d}" for index in range(12)
    }
    assert len(env.obstacle_positions) == 12
    assert len(set(env.obstacle_registry.values())) == 12
    assert set(env.obstacle_registry.values()) == env.obstacle_positions

    new_position = env.obstacle_registry[obstacle_id]
    distance = (
        abs(new_position[0] - old_position[0])
        + abs(new_position[1] - old_position[1])
    )

    assert new_position != old_position
    assert distance == 1
    assert new_position not in occupied


def test_obstacle_keeps_identity_when_walls_are_regenerated():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    obstacle_id = "O00"
    old_position = env.obstacle_registry[obstacle_id]
    occupied = {old_position}

    env.update_walls(occupied_cells=occupied)

    assert set(env.obstacle_registry) == {
        f"O{index:02d}" for index in range(12)
    }
    assert len(env.obstacle_positions) == 12
    assert len(set(env.obstacle_registry.values())) == 12
    assert set(env.obstacle_registry.values()) == env.obstacle_positions

    new_position = env.obstacle_registry[obstacle_id]
    distance = (
        abs(new_position[0] - old_position[0])
        + abs(new_position[1] - old_position[1])
    )

    assert new_position != old_position
    assert distance == 1
    assert new_position not in occupied


def test_continuous_obstacle_tracking_is_initialized():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    assert env.obstacle_trajectories == {}

    assert env.obstacle_reserved_cells == env.obstacle_positions

    assert len(env.obstacle_registry) == 12
    assert env.obstacle_positions == set(
        env.obstacle_registry.values()
    )


def test_clear_walls_only_preserves_obstacles():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    initial_registry = dict(env.obstacle_registry)
    initial_obstacle_positions = set(env.obstacle_positions)

    env._clear_walls_only()

    assert env.wall_positions == set()
    assert env.obstacle_registry == initial_registry
    assert env.obstacle_positions == initial_obstacle_positions

    for x, y in initial_obstacle_positions:
        assert maze.grid[y, x] == MOVING_OBSTACLE


def test_wall_update_preserves_active_obstacle_trajectories():
    from environment.moving_obstacle import MovingObstacle

    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    obstacle_id = "O00"
    start = env.obstacle_registry[obstacle_id]

    target = next(
        cell
        for cell in (
            (start[0] + 1, start[1]),
            (start[0] - 1, start[1]),
            (start[0], start[1] + 1),
            (start[0], start[1] - 1),
        )
        if maze.is_inside(*cell)
        and maze.grid[cell[1], cell[0]] == FREE
        and cell not in env.protected_cells
    )

    trajectory = MovingObstacle(
        start_position=(float(start[0]), float(start[1])),
        target_position=(float(target[0]), float(target[1])),
        start_time=1.0,
        duration=0.35,
    )

    env.obstacle_trajectories[obstacle_id] = trajectory
    env.obstacle_reserved_cells.update((start, target))

    env.update_walls()

    assert env.obstacle_trajectories[obstacle_id] == trajectory
    assert {start, target}.issubset(env.obstacle_reserved_cells)
    assert len(env.wall_positions) == 15
    assert len(env.obstacle_registry) == 12


def test_spawn_missing_obstacles_respects_reserved_cells():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    obstacle_id = "O00"
    old_position = env.obstacle_registry.pop(obstacle_id)
    env.obstacle_positions.remove(old_position)
    maze.grid[old_position[1], old_position[0]] = FREE

    env.obstacle_reserved_cells.add(old_position)

    env._spawn_missing_obstacles(set())

    assert obstacle_id in env.obstacle_registry
    assert env.obstacle_registry[obstacle_id] != old_position
    assert old_position not in env.obstacle_positions
    assert old_position in env.obstacle_reserved_cells


def test_direct_obstacle_update_synchronizes_reserved_cells():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    env.update_obstacles()

    assert env.obstacle_reserved_cells == env.obstacle_positions
    assert len(env.obstacle_reserved_cells) == 12
    assert set(env.obstacle_registry.values()) == env.obstacle_reserved_cells


def test_wall_update_synchronizes_reserved_cells_after_relocation():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    obstacle_id = "O00"
    old_position = env.obstacle_registry[obstacle_id]

    env.update_walls(occupied_cells={old_position})

    new_position = env.obstacle_registry[obstacle_id]

    assert new_position != old_position
    assert old_position not in env.obstacle_reserved_cells
    assert new_position in env.obstacle_reserved_cells
    assert env.obstacle_reserved_cells == env.obstacle_positions
    assert len(env.obstacle_reserved_cells) == 12


def test_update_starts_continuous_obstacle_trajectories():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    result = env.update(0.35)

    assert result["obstacles_updated"] is True
    assert env.last_obstacle_update == 0.35
    assert env.next_obstacle_update == 0.70
    assert len(env.obstacle_registry) == 12
    assert len(env.obstacle_positions) == 12

    for obstacle_id, trajectory in env.obstacle_trajectories.items():
        assert obstacle_id in env.obstacle_registry
        assert trajectory.start_time == 0.35
        assert trajectory.duration == 0.35

        source = tuple(int(round(value)) for value in trajectory.start_position)
        target = tuple(int(round(value)) for value in trajectory.target_position)

        assert source in env.obstacle_reserved_cells
        assert target in env.obstacle_reserved_cells

        for x, y in (source, target):
            assert maze.grid[y, x] == MOVING_OBSTACLE


def test_continuous_obstacle_positions_are_interpolated_between_updates():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    env.update(0.35)

    assert env.obstacle_trajectories

    obstacle_id = next(iter(env.obstacle_trajectories))
    trajectory = env.obstacle_trajectories[obstacle_id]

    midpoint_time = 0.525
    midpoint = trajectory.position_at(midpoint_time)

    expected_midpoint = tuple(
        (start + target) / 2.0
        for start, target in zip(
            trajectory.start_position,
            trajectory.target_position,
        )
    )

    assert midpoint == expected_midpoint

    previous_trajectory = trajectory
    result = env.update(midpoint_time)

    assert result["obstacles_updated"] is False
    assert env.obstacle_trajectories[obstacle_id] == previous_trajectory


def test_continuous_obstacle_scheduler_processes_all_elapsed_deadlines():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    result = env.update(1.05)

    assert result["obstacles_updated"] is True
    assert env.last_obstacle_update == 1.05
    assert env.next_obstacle_update == 1.40

    for trajectory in env.obstacle_trajectories.values():
        assert trajectory.start_time == 1.05
        assert trajectory.duration == 0.35

def test_wall_scheduler_preserves_theoretical_deadlines():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    first = env.update(5.25)

    assert first["walls_updated"] is True
    assert env.last_wall_update == 5.0
    assert env.next_wall_update == 10.0

    before_deadline = env.update(9.99)

    assert before_deadline["walls_updated"] is False
    assert env.last_wall_update == 5.0
    assert env.next_wall_update == 10.0

    second = env.update(10.25)

    assert second["walls_updated"] is True
    assert env.last_wall_update == 10.0
    assert env.next_wall_update == 15.0


def test_wall_scheduler_catches_up_after_multiple_elapsed_deadlines():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    result = env.update(16.2)

    assert result["walls_updated"] is True
    assert env.last_wall_update == 15.0
    assert env.next_wall_update == 20.0

    # The next update must still follow the theoretical schedule.
    result = env.update(19.99)

    assert result["walls_updated"] is False
    assert env.last_wall_update == 15.0
    assert env.next_wall_update == 20.0



def test_update_preserves_all_trajectories_across_elapsed_deadlines():
    from collections import Counter

    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    result = env.update(1.05)
    trajectories = result["obstacle_trajectories"]

    assert len(trajectories) == 36
    assert dict(Counter(t.start_time for t in trajectories)) == {
        0.35: 12,
        0.70: 12,
        1.05: 12,
    }
    assert len(env.obstacle_trajectories) == 12

    # Completed trajectories must remain available in interval history.
    assert sum(t.end_time <= 1.05 for t in trajectories) == 24


def test_update_history_does_not_reuse_previous_interval_trajectories():
    maze = Maze(width=25, height=25, seed=42)
    env = DynamicEnvironment(maze, seed=42)

    first_result = env.update(1.05)
    second_result = env.update(1.10)

    first_history = first_result["obstacle_trajectories"]
    second_history = second_result["obstacle_trajectories"]

    assert len(first_history) == 36
    assert len(second_history) == 12
    assert all(t.start_time == 1.05 for t in second_history)
    # Trajectories still active at the interval boundary must be reused.
    assert any(
        previous is current
        for previous in first_history
        for current in second_history
    )

    # Trajectories completed before the second interval must not reappear.
    assert all(t.start_time >= 1.05 for t in second_history)
