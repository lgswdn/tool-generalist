import importlib
import sys


def _new_modules(before):
    return set(sys.modules).difference(before)


def _has_prefix(modules, prefix):
    return any(name == prefix or name.startswith(prefix + ".") for name in modules)


def test_get_isaac_runner_is_lazy_and_does_not_import_isaac_modules():
    before = set(sys.modules)

    physics = importlib.import_module("utils.contact.stabilize")
    runner = physics.get_physics_runner("isaac")

    assert runner.name == "isaac"
    created = _new_modules(before)
    assert "utils.contact.isaac" not in created
    assert not _has_prefix(created, "isaacsim")
    assert not _has_prefix(created, "omni")
    assert not _has_prefix(created, "pxr")


def test_real_adapter_module_import_is_safe_without_simulation_app():
    before = set(sys.modules)

    module = importlib.import_module("utils.contact.isaac")

    assert hasattr(module, "IsaacSimAdapter")
    created = _new_modules(before)
    assert not _has_prefix(created, "isaacsim")
    assert not _has_prefix(created, "omni")
    assert not _has_prefix(created, "pxr")
