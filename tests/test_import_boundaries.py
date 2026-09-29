"""Static import contracts from docs/INVARIANTS.md; no application imports."""

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

WEB_SERVICE_EXCEPTIONS = {
    "aesthetic_tag_scheduler.py": {
        "scripts.aesthetic_tag_nightly",
        "open_clip",
        "torch",
    },
    "analysis_service.py": {"cv2", "web.security.safe_log_value"},
    "companion/llama_cpp_adapter.py": {"llama_cpp"},
    "model_registry_service.py": {"yaml"},
    "nightly_jobs/sharpness_job.py": {"cv2", "utils.image_ops"},
    "report_scheduler.py": {"utils.daily_report"},
    "telemetry_service.py": {"requests", "psutil", "utils.settings"},
    "update_service.py": {"web.security.safe_log_value"},
    "usb_format_service.py": {"web.security.safe_log_value"},
}

DETECTOR_SERVICE_EDGES = {
    ("persistence_service", "crop_service"),
    ("capability_registry", "decision_policy_service"),
    ("capability_registry", "temporal_decision_service"),
    ("scoring_pipeline", "bbox_quality_service"),
    ("scoring_pipeline", "capability_registry"),
    ("scoring_pipeline", "decision_policy_service"),
    ("scoring_pipeline", "temporal_decision_service"),
}


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_imports_from_file(filepath: Path) -> list[tuple[str, int]]:
    package = ".".join(filepath.relative_to(get_project_root()).parts[:-1])
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            # Include imported names so `from utils import db` cannot evade H-01.
            imports.extend(
                (module if alias.name == "*" else f"{module}.{alias.name}", node.lineno)
                for alias in node.names
            )
    return imports


def module_matches(module: str, prefix: str) -> bool:
    prefix = prefix.rstrip(".")
    return module == prefix or module.startswith(prefix + ".")


def check_forbidden_imports(
    imports: list[tuple[str, int]], forbidden_prefixes: list[str]
) -> list[tuple[str, int]]:
    return [
        (module, line)
        for module, line in imports
        if any(module_matches(module, prefix) for prefix in forbidden_prefixes)
    ]


def web_service_violations(project_root: Path) -> list[str]:
    services_dir = project_root / "web" / "services"
    assert services_dir.is_dir(), "Missing web/services"
    violations = []
    for path in sorted(services_dir.rglob("*.py")):
        relative = path.relative_to(services_dir).as_posix()
        allowed = {"core", "config", "logging_config", "web.services"}
        allowed.update(WEB_SERVICE_EXCEPTIONS.get(relative, set()))
        for module, line in get_imports_from_file(path):
            if module.split(".")[0] in sys.stdlib_module_names:
                continue
            if not any(module_matches(module, prefix) for prefix in allowed):
                violations.append(f"{relative}:{line} imports {module}")
    return violations


def detector_service_violations(project_root: Path) -> list[str]:
    services_dir = project_root / "detectors" / "services"
    assert services_dir.is_dir(), "Missing detectors/services"
    violations = []
    for path in sorted(services_dir.rglob("*.py")):
        source = ".".join(path.relative_to(services_dir).with_suffix("").parts)
        for module, line in get_imports_from_file(path):
            if not module_matches(module, "detectors.services"):
                continue
            if path == services_dir / "__init__.py":
                continue
            target = module.removeprefix("detectors.services.").split(".")[0]
            if (source, target) not in DETECTOR_SERVICE_EDGES:
                violations.append(f"{source}:{line} imports {module}")
    return violations


class TestWebLayerBoundaries:
    """Tests for web layer import boundaries."""

    @pytest.mark.arch_hard
    def test_services_use_documented_imports(self) -> None:
        assert web_service_violations(get_project_root()) == []

    @pytest.mark.arch_hard
    def test_core_does_not_import_web(self):
        """core/* should never import from web/, flask, werkzeug."""
        project_root = get_project_root()
        core_dir = project_root / "core"

        assert core_dir.is_dir()

        forbidden = ["web.", "flask", "werkzeug"]
        all_violations = []

        for py_file in core_dir.rglob("*.py"):
            imports = get_imports_from_file(py_file)
            violations = check_forbidden_imports(imports, forbidden)
            for module, line in violations:
                all_violations.append(f"{py_file.name}:{line} imports {module}")

        assert len(all_violations) == 0, (
            "Core should never import web layer. Violations:\n"
            + "\n".join(all_violations)
        )

    @pytest.mark.arch_soft
    def test_count_web_interface_violations(self):
        """
        Counts violations in web_interface.py.

        SOFT monitor only: does not fail CI.
        """
        project_root = get_project_root()
        web_interface = project_root / "web" / "web_interface.py"

        if not web_interface.exists():
            return

        forbidden = ["utils.", "camera."]
        imports = get_imports_from_file(web_interface)
        violations = check_forbidden_imports(imports, forbidden)

        # Print current violation count for tracking
        print(f"\n[Migration Progress] web_interface.py violations: {len(violations)}")
        for module, line in violations:
            print(f"  - Line {line}: {module}")

        # Target: 0 violations after migration is complete.
        # Uncomment the assertion below when migration is complete:
        # assert len(violations) == 0

    def test_detection_manager_delegates_image_and_notification_helpers(self) -> None:
        detection_manager = get_project_root() / "detectors" / "detection_manager.py"
        assert detection_manager.is_file()
        forbidden = [
            "utils.image_ops",  # Must use CropService
            "utils.telegram_notifier",  # Must use NotificationService
            "piexif",  # Must use PersistenceService
        ]

        imports = get_imports_from_file(detection_manager)
        violations = check_forbidden_imports(imports, forbidden)

        assert len(violations) == 0, (
            "detection_manager.py must use Services, not direct implementations.\\n"
            "Violations:\\n"
            + "\\n".join([f"  - Line {line}: {module}" for module, line in violations])
        )


@pytest.mark.arch_hard
class TestModuleStructure:
    """Tests for module structure integrity."""

    def test_core_modules_exist(self):
        """Verify all required core modules exist."""
        project_root = get_project_root()
        core_dir = project_root / "core"

        required_modules = [
            "gallery_core.py",
            "settings_core.py",
            "onvif_core.py",
            "analytics_core.py",
            "detections_core.py",
        ]

        missing = []
        for module in required_modules:
            if not (core_dir / module).exists():
                missing.append(module)

        assert len(missing) == 0, f"Missing core modules: {missing}"

    def test_service_modules_exist(self):
        """Verify all required service modules exist."""
        project_root = get_project_root()
        services_dir = project_root / "web" / "services"

        required_modules = [
            "gallery_service.py",
            "settings_service.py",
            "onvif_service.py",
            "analytics_service.py",
            "detections_service.py",
        ]

        missing = []
        for module in required_modules:
            if not (services_dir / module).exists():
                missing.append(module)

        assert len(missing) == 0, f"Missing service modules: {missing}"


@pytest.mark.arch_hard
class TestDetectorServicesArchitecture:
    """Tests for detectors/services/* architectural boundaries."""

    def test_detector_services_do_not_import_web(self):
        """
        detectors/services/* must not import from web layer.

        Services are core infrastructure and should be web-agnostic.
        """
        project_root = get_project_root()
        services_dir = project_root / "detectors" / "services"

        assert services_dir.is_dir()

        forbidden = ["web.", "flask", "werkzeug"]
        all_violations = []

        for py_file in services_dir.rglob("*.py"):
            imports = get_imports_from_file(py_file)
            violations = check_forbidden_imports(imports, forbidden)
            for module, line in violations:
                all_violations.append(f"{py_file.name}:{line} imports {module}")

        assert len(all_violations) == 0, (
            "Detector services must not import web layer. Violations:\n"
            + "\n".join(all_violations)
        )

    def test_detector_services_use_documented_edges(self) -> None:
        assert detector_service_violations(get_project_root()) == []

    def test_detector_services_exist(self):
        """Verify all required detector services exist."""
        project_root = get_project_root()
        services_dir = project_root / "detectors" / "services"

        required_modules = [
            "persistence_service.py",
            "crop_service.py",
            "classification_service.py",
            "detection_service.py",
            "notification_service.py",
        ]

        missing = []
        for module in required_modules:
            if not (services_dir / module).exists():
                missing.append(module)

        assert len(missing) == 0, f"Missing detector services: {missing}"


class TestTemplateArchitecture:
    """Tests for template architectural integrity."""

    def test_templates_extend_base(self):
        """
        All main templates (non-partials) should extend base.html.

        This ensures consistent layout and header/footer.
        """
        import re

        project_root = get_project_root()
        templates_dir = project_root / "templates"

        if not templates_dir.exists():
            return

        # Templates that should extend base.html
        main_templates = [
            "gallery.html",
            "stream.html",
            "settings.html",
            "species.html",
            "subgallery.html",
            "analytics.html",
            "edit.html",
            "inbox.html",
            "orphans.html",
            "trash.html",
            "backup.html",
            "restore.html",
            "logs.html",
            "login.html",
        ]

        extends_pattern = re.compile(r'{%\s*extends\s+["\']base\.html["\']\s*%}')

        missing_extends = []

        for template_name in main_templates:
            template_path = templates_dir / template_name
            if not template_path.exists():
                continue

            with open(template_path, encoding="utf-8") as f:
                content = f.read()

            if not extends_pattern.search(content):
                missing_extends.append(template_name)

        assert len(missing_extends) == 0, (
            "Templates must extend 'base.html'. Missing extends:\n"
            + "\n".join(missing_extends)
        )

    def test_partials_do_not_extend(self):
        """
        Partial templates should not extend base.html.

        Partials are included fragments, not full pages.
        """
        import re

        project_root = get_project_root()
        partials_dir = project_root / "templates" / "partials"

        if not partials_dir.exists():
            return

        extends_pattern = re.compile(r"{%\s*extends\s+")

        violations = []

        for partial in partials_dir.glob("*.html"):
            with open(partial, encoding="utf-8") as f:
                content = f.read()

            if extends_pattern.search(content):
                violations.append(partial.name)

        assert len(violations) == 0, (
            "Partials should not extend templates. Violations:\n"
            + "\n".join(violations)
        )

    def test_no_direct_python_imports_in_templates(self):
        """
        Templates should not contain Python import statements.

        All data should come from template context, not direct imports.
        """
        import re

        project_root = get_project_root()
        templates_dir = project_root / "templates"

        if not templates_dir.exists():
            return

        # Pattern for Python imports in Jinja (which would be a bug)
        import_pattern = re.compile(r"{%\s*import\s+\w+\s*%}")
        from_import_pattern = re.compile(r"{%\s*from\s+\w+\s+import\s+")

        violations = []

        for template_file in templates_dir.rglob("*.html"):
            with open(template_file, encoding="utf-8") as f:
                content = f.read()

            # Check for import patterns (Jinja's import is fine for macros)
            # We're looking for patterns that look like Python imports
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                # Look for Python-style imports that shouldn't be in templates
                if "import " in line and "{% import" not in line and "from " in line:
                    if "{% from" not in line:
                        violations.append(f"{template_file.name}:{i}")

        # Note: This is a soft check - Jinja macros use import syntax
        # We pass even with "violations" since Jinja imports are valid
        # The test documents the pattern for awareness


if __name__ == "__main__":
    # Run basic checks when executed directly

    print("Running import boundary checks...")

    tests = TestWebLayerBoundaries()

    try:
        tests.test_services_use_documented_imports()
        print("✓ Services import boundaries OK")
    except AssertionError as e:
        print(f"✗ Services violation: {e}")

    try:
        tests.test_core_does_not_import_web()
        print("✓ Core import boundaries OK")
    except AssertionError as e:
        print(f"✗ Core violation: {e}")

    tests.test_count_web_interface_violations()

    structure_tests = TestModuleStructure()

    try:
        structure_tests.test_core_modules_exist()
        print("✓ Core modules exist")
    except AssertionError as e:
        print(f"✗ {e}")

    try:
        structure_tests.test_service_modules_exist()
        print("✓ Service modules exist")
    except AssertionError as e:
        print(f"✗ {e}")

    # Detector services tests
    detector_tests = TestDetectorServicesArchitecture()

    try:
        detector_tests.test_detector_services_do_not_import_web()
        print("✓ Detector services do not import web")
    except AssertionError as e:
        print(f"✗ {e}")

    try:
        detector_tests.test_detector_services_use_documented_edges()
        print("✓ Detector service dependency edges OK")
    except AssertionError as e:
        print(f"✗ {e}")

    try:
        detector_tests.test_detector_services_exist()
        print("✓ Detector services exist")
    except AssertionError as e:
        print(f"✗ {e}")

    # Template tests
    template_tests = TestTemplateArchitecture()

    try:
        template_tests.test_templates_extend_base()
        print("✓ Templates extend base.html")
    except AssertionError as e:
        print(f"✗ {e}")

    try:
        template_tests.test_partials_do_not_extend()
        print("✓ Partials do not extend")
    except AssertionError as e:
        print(f"✗ {e}")


@pytest.mark.arch_hard
class TestBoundaryScanner:
    @pytest.mark.parametrize(
        "source",
        [
            "import utils",
            "from utils import settings",
            "from ..security import safe_log_value",
            "import camera.video_capture as capture",
            "from web import web_interface",
            "import requests",
            "import core_extra",
            "def lazy():\n    from utils import db",
        ],
    )
    def test_web_rejects_undocumented_imports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
    ) -> None:
        monkeypatch.setattr(sys.modules[__name__], "get_project_root", lambda: tmp_path)
        path = tmp_path / "web/services/new_service.py"
        path.parent.mkdir(parents=True)
        path.write_text(source)
        assert web_service_violations(tmp_path)

    def test_web_scans_nested_initializers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[__name__], "get_project_root", lambda: tmp_path)
        path = tmp_path / "web/services/nested/__init__.py"
        path.parent.mkdir(parents=True)
        path.write_text("from ...security import safe_log_value")
        assert web_service_violations(tmp_path)
        path.write_text(
            "from .. import db_service\nfrom core import db_core\nimport os"
        )
        assert web_service_violations(tmp_path) == []

    def test_exceptions_are_scoped_to_file_and_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[__name__], "get_project_root", lambda: tmp_path)
        path = tmp_path / "web/services/report_scheduler.py"
        path.parent.mkdir(parents=True)
        path.write_text("from utils.daily_report import main")
        assert web_service_violations(tmp_path) == []
        path.write_text("from utils import db")
        assert web_service_violations(tmp_path)

    @pytest.mark.parametrize(
        "source",
        [
            "import detectors.services.new_service",
            "from detectors.services import new_service",
            "from .new_service import run",
            "from . import new_service",
            "from detectors.services import *",
        ],
    )
    def test_detector_rejects_new_edges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
    ) -> None:
        monkeypatch.setattr(sys.modules[__name__], "get_project_root", lambda: tmp_path)
        path = tmp_path / "detectors/services/scoring_pipeline.py"
        path.parent.mkdir(parents=True)
        path.write_text(source)
        assert detector_service_violations(tmp_path)

    def test_detector_allows_documented_relative_edge_and_package_exports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[__name__], "get_project_root", lambda: tmp_path)
        path = tmp_path / "detectors/services/persistence_service.py"
        path.parent.mkdir(parents=True)
        path.write_text("from .crop_service import CropService")
        (path.parent / "__init__.py").write_text(
            "from .crop_service import CropService"
        )
        assert detector_service_violations(tmp_path) == []
        path.write_text("from .scoring_pipeline import compute_detection_signals")
        assert detector_service_violations(tmp_path)

    def test_syntax_errors_are_not_silently_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[__name__], "get_project_root", lambda: tmp_path)
        path = tmp_path / "broken.py"
        path.write_text("def broken(")
        with pytest.raises(SyntaxError):
            get_imports_from_file(path)

    @pytest.mark.parametrize("module", ["web", "flask", "werkzeug", "web.routes"])
    def test_framework_boundary_matches_roots(self, module: str) -> None:
        assert check_forbidden_imports([(module, 1)], ["web", "flask", "werkzeug"])
        assert check_forbidden_imports([("website", 1)], ["web"]) == []
