"""examples/random_agents.py runs end to end: one env, a rendered GIF, and the vector env."""
import importlib.util
from pathlib import Path

import pytest

from foraging_env import HELDOUT_SEEDS

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "random_agents.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("random_agents", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_env(example):
    results = example.main(["--episodes", "2", "--max-steps", "30", "--no-render"])
    assert [r["layout_seed"] for r in results] == list(HELDOUT_SEEDS[:2])
    assert all(r["steps"] == 30 and r["truncated"] for r in results)
    assert results == example.main(["--episodes", "2", "--max-steps", "30", "--no-render"])   # reproducible


def test_rendered_episode_gif(example, tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "episode.gif"
    results = example.main(["--level", "L2", "--episodes", "1", "--max-steps", "20", "--gif", str(path)])
    assert len(results) == 1 and path.stat().st_size > 0


def test_vector_env(example):
    results = example.main(["--n-envs", "2", "--episodes", "2", "--max-steps", "25", "--no-render"])
    assert len(results) == 2 and all(r["steps"] == 25 for r in results)
    assert sorted(r["layout_seed"] for r in results) == list(HELDOUT_SEEDS[:2])
