"""Deterministic per-env tool/object asset assignment helpers."""

from __future__ import annotations

import random


TOOL_ASSIGNMENT_SALT = 17
OBJECT_ASSIGNMENT_SALT = 53
GENERATED_GRIPPER_ASSIGNMENT_SALT = 71
ONE_DOF_GRIPPER_ASSIGNMENT_SALT = 89


def cross_embodiment_mode_for_rank(global_rank: int, world_size: int) -> str:
    """Assign exactly half of an even distributed world to each gripper topology."""

    global_rank = _require_non_negative_int("global_rank", global_rank)
    world_size = _require_positive_int("world_size", world_size)
    if world_size < 2 or world_size % 2 != 0:
        raise ValueError("cross-embodiment assignment requires an even world_size >= 2")
    if global_rank >= world_size:
        raise ValueError("global_rank must be smaller than world_size")
    return "generated_gripper" if global_rank < world_size // 2 else "one_dof_gripper"


def global_env_id(local_env_id: int, num_envs_per_rank: int, global_rank: int) -> int:
    """Return the global env id for a rank-local env index."""

    local_env_id = _require_int("local_env_id", local_env_id)
    num_envs_per_rank = _require_positive_int("num_envs_per_rank", num_envs_per_rank)
    global_rank = _require_non_negative_int("global_rank", global_rank)
    if local_env_id < 0 or local_env_id >= num_envs_per_rank:
        raise ValueError("local_env_id must be in [0, num_envs_per_rank)")
    return global_rank * num_envs_per_rank + local_env_id


def asset_index_for_env(
    local_env_id: int,
    num_envs_per_rank: int,
    global_rank: int,
    num_assets: int,
    *,
    randomize: bool,
    seed: int,
    salt: int,
) -> int:
    """Return the asset index assigned to one rank-local env."""

    num_assets = _require_positive_int("num_assets", num_assets)
    seed = _require_non_negative_int("seed", seed)
    salt = _require_int("salt", salt)
    if not isinstance(randomize, bool):
        raise ValueError("randomize must be a bool")

    env_id = global_env_id(local_env_id, num_envs_per_rank, global_rank)
    if not randomize:
        return env_id % num_assets
    rng = random.Random(seed + env_id * 1000003 + salt)
    return rng.randrange(num_assets)


def asset_indices_for_rank(
    num_envs_per_rank: int,
    global_rank: int,
    num_assets: int,
    *,
    randomize: bool,
    seed: int,
    salt: int,
) -> list[int]:
    """Return asset indices for every local env on one rank."""

    num_envs_per_rank = _require_positive_int("num_envs_per_rank", num_envs_per_rank)
    global_rank = _require_non_negative_int("global_rank", global_rank)
    num_assets = _require_positive_int("num_assets", num_assets)
    return [
        asset_index_for_env(
            local_env_id,
            num_envs_per_rank,
            global_rank,
            num_assets,
            randomize=randomize,
            seed=seed,
            salt=salt,
        )
        for local_env_id in range(num_envs_per_rank)
    ]


def sequential_spawn_indices_for_rank(
    num_envs_per_rank: int,
    global_rank: int,
    num_assets: int,
) -> list[int]:
    """Return the compact spawn list for deterministic modulo assignment.

    Isaac Lab's MultiAssetSpawnerCfg selects prototype ``local_env_id % len(assets_cfg)``
    when ``random_choice=False``.  For deterministic assignment, the per-env sequence is
    a rank-offset modulo cycle, so we only need one cycle of prototypes rather than one
    entry per environment.
    """

    num_envs_per_rank = _require_positive_int("num_envs_per_rank", num_envs_per_rank)
    global_rank = _require_non_negative_int("global_rank", global_rank)
    num_assets = _require_positive_int("num_assets", num_assets)
    start = (global_rank * num_envs_per_rank) % num_assets
    count = min(num_envs_per_rank, num_assets)
    return [(start + local_index) % num_assets for local_index in range(count)]


def _require_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    value = _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_positive_int(name: str, value: int) -> int:
    value = _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value
