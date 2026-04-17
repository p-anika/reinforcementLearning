"""
env/potential.py
----------------
Environment-specific potential functions for the Bayesian reality-check mechanism.

Each class implements get_potential(env) → float, where env is the unwrapped
MiniGrid environment (env.unwrapped). Higher potential = closer to task completion.
All potentials are non-positive, reaching 0 only at the goal.

The potential difference Δφ = φ(s_{t+1}) − φ(s_t) is used to verify whether
the human teacher's feedback aligns with the agent's actual progress.

Supported environments:
    EmptyPotential    → MiniGrid-Empty-8x8-v0
    LavaGapPotential  → MiniGrid-LavaGap-7x7-v0
    DoorKeyPotential  → MiniGrid-DoorKey-8x8-v0

Factory:
    get_potential_fn(env_id) → BasePotential instance
"""

import numpy as np
from abc import ABC, abstractmethod


# ─── Abstract Base ────────────────────────────────────────────────────────────

class BasePotential(ABC):
    """Abstract interface for environment-specific potential functions."""

    @abstractmethod
    def get_potential(self, env):
        """
        Compute the potential φ(s) for the agent's current state.

        Parameters
        ----------
        env : MiniGridEnv  the unwrapped base environment (env.unwrapped)

        Returns
        -------
        float : potential value (≤ 0; higher means closer to goal)
        """
        pass

    def _manhattan(self, pos_a, pos_b):
        """Manhattan distance between two (x, y) positions."""
        return int(abs(int(pos_a[0]) - int(pos_b[0])) + abs(int(pos_a[1]) - int(pos_b[1])))

    def _find_tile(self, env, tile_type):
        """
        Scan the grid and return the (x, y) position of the first tile of
        the given type, or None if not found.
        """
        grid = env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                tile = grid.get(x, y)
                if tile and tile.type == tile_type:
                    return np.array([x, y])
        return None


# ─── Environment-Specific Implementations ─────────────────────────────────────

class EmptyPotential(BasePotential):
    """
    MiniGrid-Empty-8x8-v0 potential.

    φ = −manhattan_distance(agent, goal)

    Monotonically increases as the agent approaches the goal.
    Reaches 0 when the agent is at the goal tile.
    """

    def get_potential(self, env):
        goal_pos = self._find_tile(env, "goal")
        if goal_pos is None:
            return 0.0
        return -float(self._manhattan(env.agent_pos, goal_pos))


class LavaGapPotential(BasePotential):
    """
    MiniGrid-LavaGap-7x7-v0 potential.

    φ = −manhattan_distance(agent, goal) − lava_penalty

    lava_penalty = 1 if the agent is orthogonally adjacent to any lava tile,
                   else 0.

    The distance term drives progress toward the goal; the lava_penalty
    discourages the human from giving positive feedback when the agent is
    in danger (adjacent to lava), since moving toward lava would actually
    be a bad step even if it reduces distance to the goal.
    """

    def get_potential(self, env):
        goal_pos = self._find_tile(env, "goal")
        if goal_pos is None:
            return 0.0

        dist = self._manhattan(env.agent_pos, goal_pos)

        # Check the four orthogonal neighbours for lava tiles
        ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
        neighbors = [(ax - 1, ay), (ax + 1, ay), (ax, ay - 1), (ax, ay + 1)]
        lava_penalty = 0
        for nx, ny in neighbors:
            if 0 <= nx < env.grid.width and 0 <= ny < env.grid.height:
                tile = env.grid.get(nx, ny)
                if tile and tile.type == "lava":
                    lava_penalty = 1
                    break

        return -float(dist) - float(lava_penalty)


class DoorKeyPotential(BasePotential):
    """
    MiniGrid-DoorKey-8x8-v0 staged potential.

    The mission has three sequential sub-phases. The potential function
    switches phase based on the agent's inventory and the door state:

        Phase 1 (no key held):   φ = −distance(agent, key)
        Phase 2 (key held):      φ = −distance(agent, door)
        Phase 3 (door open):     φ = −distance(agent, goal)

    This guides the human teacher's feedback through the full key → door → goal
    sequence rather than conflating all three sub-tasks into one signal.
    """

    def get_potential(self, env):
        agent_pos = env.agent_pos

        # ── Phase 3: door is open → navigate to goal ──────────────────────────
        door_pos = self._find_tile(env, "door")
        if door_pos is not None:
            door_tile = env.grid.get(int(door_pos[0]), int(door_pos[1]))
            if door_tile and getattr(door_tile, "is_open", False):
                goal_pos = self._find_tile(env, "goal")
                if goal_pos is None:
                    return 0.0
                return -float(self._manhattan(agent_pos, goal_pos))

        # ── Phase 2: agent carries a key → navigate to door ───────────────────
        if env.carrying is not None and env.carrying.type == "key":
            if door_pos is None:
                return 0.0
            return -float(self._manhattan(agent_pos, door_pos))

        # ── Phase 1: no key held → find the key first ─────────────────────────
        key_pos = self._find_tile(env, "key")
        if key_pos is None:
            return 0.0
        return -float(self._manhattan(agent_pos, key_pos))


# ─── Factory ──────────────────────────────────────────────────────────────────

_POTENTIAL_CLASSES = {
    "MiniGrid-Empty-8x8-v0":   EmptyPotential,
    "MiniGrid-LavaGapS7-v0": LavaGapPotential,
    "MiniGrid-DoorKey-8x8-v0": DoorKeyPotential,
}


def get_potential_fn(env_id):
    """
    Return an instantiated potential function for the given environment ID.

    Raises ValueError for unknown env_ids so errors are caught early
    rather than silently falling back to a wrong potential.
    """
    if env_id not in _POTENTIAL_CLASSES:
        raise ValueError(
            f"No potential function registered for '{env_id}'. "
            f"Available: {list(_POTENTIAL_CLASSES)}"
        )
    return _POTENTIAL_CLASSES[env_id]()
