import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_GENERATOR = "contact" + "_gen" + "_new"
OLD_BATCH = "gen" + "_dataset"
OLD_CONFIG = "contact" + "_config"


def _top_level_api_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_old_contact_entrypoints_are_removed_or_hard_error():
    old_files = {
        ROOT / "pretrain" / "contact_gen.py": "contact_generation.generator",
        ROOT / "pretrain" / "gen_dataset.py": "contact_generation.batch_generate",
        ROOT / "pretrain" / "contact_config.py": "contact_generation.config",
        ROOT / "pretrain" / "gen_initial.py": "contact_generation.generator",
        ROOT / "pretrain" / "gen_movement_delta.py": "contact_generation.generator",
        ROOT / "pretrain" / "validate_contact_physics.py": "contact_generation",
        ROOT / "pretrain" / "corn.py": "contact_generation",
        ROOT / "pretrain" / "new_pretrain" / f"{OLD_GENERATOR}.py": "contact_generation.generator",
        ROOT / "pretrain" / "new_pretrain" / f"{OLD_BATCH}.py": "contact_generation.batch_generate",
        ROOT / "pretrain" / "new_pretrain" / f"{OLD_CONFIG}.py": "contact_generation.config",
        ROOT / "pretrain" / "new_pretrain" / "corn_dataset.py": "contact_generation",
    }
    forbidden_public_api = {
        "Config",
        "ContactGenHyperparams",
        "InitialPoseHyperparams",
        "MovementDeltaHyperparams",
        "main",
        "run_pair",
        "CONTACT_GEN",
        "INITIAL_POSE",
        "MOVEMENT_DELTA",
    }

    for path, replacement in old_files.items():
        if not path.exists():
            continue

        text = path.read_text()
        assert replacement in text
        assert any(marker in text for marker in ("raise ", "RuntimeError", "SystemExit", "sys.exit"))
        assert not (_top_level_api_names(path) & forbidden_public_api)


def test_pretrain_does_not_import_canonical_contact_generation_entrypoints():
    forbidden_modules = {
        "contact_generation.generator",
        "contact_generation.batch_generate",
    }
    forbidden_from_contact_generation = {"generator", "batch_generate"}
    vendored_roots = {"diffusion_policy_repo"}
    violations: list[str] = []

    for path in (ROOT / "pretrain").rglob("*.py"):
        rel = path.relative_to(ROOT / "pretrain")
        if rel.parts and rel.parts[0] in vendored_roots:
            continue
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            for lineno, line in enumerate(source.splitlines(), start=1):
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                for module in forbidden_modules:
                    if module in stripped:
                        violations.append(f"{path}:{lineno}: {stripped}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_modules:
                    violations.append(f"{path}: from {module} import ...")
                if module == "contact_generation":
                    for alias in node.names:
                        if alias.name in forbidden_from_contact_generation:
                            violations.append(f"{path}: from contact_generation import {alias.name}")
            elif isinstance(node, ast.Call):
                func = node.func
                is_dynamic_import = (
                    isinstance(func, ast.Name)
                    and func.id in {"__import__", "import_module"}
                ) or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "import_module"
                )
                if is_dynamic_import and node.args and isinstance(node.args[0], ast.Constant):
                    if node.args[0].value in forbidden_modules:
                        violations.append(f"{path}: dynamic import {node.args[0].value}")

    assert violations == []


def test_legacy_corn_pickle_loader_is_removed():
    path = ROOT / "pretrain" / "new_pretrain" / "corn_dataset.py"
    assert not path.exists()


def test_new_pretrain_train_entrypoint_is_removed():
    train_py = ROOT / "pretrain" / "new_pretrain" / "train.py"
    assert not train_py.exists()


def test_tests_do_not_import_removed_contact_entrypoints():
    forbidden_refs = {
        f"pretrain.new_pretrain.{OLD_GENERATOR}",
        f"pretrain.new_pretrain.{OLD_BATCH}",
        f"new_pretrain/{OLD_GENERATOR}.py",
        f"new_pretrain/{OLD_BATCH}.py",
        OLD_GENERATOR,
    }
    violations: list[str] = []
    this_file = Path(__file__).resolve()

    for path in (ROOT / "tests").rglob("*.py"):
        if path.resolve() == this_file:
            continue
        text = path.read_text()
        for ref in forbidden_refs:
            if ref in text:
                violations.append(f"{path}: {ref}")

    assert violations == []
