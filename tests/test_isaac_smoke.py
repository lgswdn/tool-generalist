import os

import pytest


pytestmark = pytest.mark.isaac


@pytest.mark.skipif(os.environ.get("RUN_ISAAC_SMOKE") != "1", reason="set RUN_ISAAC_SMOKE=1 to launch Isaac Sim")
def test_isaac_adapter_can_launch_and_close_simulation_app():
    from utils.contact.isaac import IsaacAdapterUnavailable, IsaacSimAdapter

    adapter = IsaacSimAdapter(headless=True)
    try:
        try:
            adapter.initialize()
        except IsaacAdapterUnavailable as exc:
            pytest.skip(str(exc))
        assert adapter.is_real_physics is True
    finally:
        adapter.close()
