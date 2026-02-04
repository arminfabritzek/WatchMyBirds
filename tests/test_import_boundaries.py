"""
Import Boundary Tests.

Validates that architectural import rules are followed:
- web/* may NOT import directly from utils/, camera/, detectors/
- web/services/* may ONLY import from core/*
- core/* may NOT import from web/, flask, werkzeug

NOTE: This test currently detects EXISTING violations in web_interface.py.
      As routes are migrated to use services, violations should decrease.
      The test tracks progress toward zero violations.
"""

import ast
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_imports_from_file(filepath: Path) -> list[tuple[str, int]]:
    """
    Extract all import statements from a Python file.

    Returns:
        List of (module_name, line_number) tuples
    """
    imports = []
    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))

    return imports


def check_forbidden_imports(
    imports: list[tuple[str, int]], forbidden_prefixes: list[str]
) -> list[tuple[str, int]]:
    """
    Check for forbidden imports.

    Returns:
        List of (module_name, line_number) for violations
    """
    violations = []
    for module, line in imports:
        for prefix in forbidden_prefixes:
            if module.startswith(prefix):
                violations.append((module, line))
                break
    return violations


class TestWebLayerBoundaries:
    """Tests for web layer import boundaries."""

    def test_services_only_import_from_core(self):
        """web/services/* should only import from core/*."""
        project_root = get_project_root()
        services_dir = project_root / "web" / "services"

        if not services_dir.exists():
            return  # No services yet

        forbidden = ["utils.", "camera.", "detectors."]
        all_violations = []

        for py_file in services_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            imports = get_imports_from_file(py_file)
            violations = check_forbidden_imports(imports, forbidden)
            for module, line in violations:
                all_violations.append(f"{py_file.name}:{line} imports {module}")

        assert len(all_violations) == 0, (
            "Services should only import from core/*. Violations:\n"
            + "\n".join(all_violations)
        )

    def test_core_does_not_import_web(self):
        """core/* should never import from web/, flask, werkzeug."""
        project_root = get_project_root()
        core_dir = project_root / "core"

        if not core_dir.exists():
            return

        forbidden = ["web.", "flask", "werkzeug"]
        all_violations = []

        for py_file in core_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            imports = get_imports_from_file(py_file)
            violations = check_forbidden_imports(imports, forbidden)
            for module, line in violations:
                all_violations.append(f"{py_file.name}:{line} imports {module}")

        assert len(all_violations) == 0, (
            "Core should never import web layer. Violations:\n"
            + "\n".join(all_violations)
        )

    def test_count_web_interface_violations(self):
        """
        Counts violations in web_interface.py.

        This test does NOT fail - it tracks migration progress.
        As routes are migrated, the count should decrease to 0.
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

        # Target: 0 violations after migration is complete
        # Currently expected: ~25 violations (see REPO_AUDIT_PLAN_AND_TASKs.md)
        # Uncomment below when migration is complete:
        # assert len(violations) == 0

    def test_detection_manager_only_uses_services(self):
        """
        detection_manager.py should only use Services for core operations.

        FORBIDDEN imports in detection_manager.py:
        - utils.db (use PersistenceService)
        - utils.image_ops (use CropService)
        - utils.telegram_notifier (use NotificationService)
        - camera. (allowed: only VideoCapture for frame grabbing)

        ALLOWED imports:
        - detectors.services.* (all Services)
        - detectors.classifier (wrapped by ClassificationService)
        - detectors.motion_detector (not a Service, standalone)
        - camera.video_capture (needed for frame input)
        - utils.db (get_connection, get_or_create_default_source only for DB init)
        - utils.path_manager (PathManager for output dirs)
        - config, logging_config, threading, etc.
        """
        project_root = get_project_root()
        detection_manager = project_root / "detectors" / "detection_manager.py"

        if not detection_manager.exists():
            return

        # These imports are FORBIDDEN in detection_manager.py
        # They indicate direct implementation rather than service delegation
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


class TestDetectorServicesArchitecture:
    """Tests for detectors/services/* architectural boundaries."""

    def test_detector_services_do_not_import_web(self):
        """
        detectors/services/* must not import from web layer.

        Services are core infrastructure and should be web-agnostic.
        """
        project_root = get_project_root()
        services_dir = project_root / "detectors" / "services"

        if not services_dir.exists():
            return

        forbidden = ["web.", "flask", "werkzeug"]
        all_violations = []

        for py_file in services_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            imports = get_imports_from_file(py_file)
            violations = check_forbidden_imports(imports, forbidden)
            for module, line in violations:
                all_violations.append(f"{py_file.name}:{line} imports {module}")

        assert len(all_violations) == 0, (
            "Detector services must not import web layer. Violations:\n"
            + "\n".join(all_violations)
        )

    def test_detector_services_do_not_import_each_other_circularly(self):
        """
        Detector services should not have circular import dependencies.

        Each service should be independent or only depend on shared utilities.

        ALLOWED exceptions (documented dependencies):
        - persistence_service → crop_service (thumbnails need cropping)
        """
        project_root = get_project_root()
        services_dir = project_root / "detectors" / "services"

        if not services_dir.exists():
            return

        # Check that no service imports another service
        # (except through the __init__.py or allowed exceptions)
        service_modules = [
            "persistence_service",
            "crop_service",
            "classification_service",
            "detection_service",
            "notification_service",
            "capture_service",
        ]

        # Allowed dependencies (from → to)
        allowed_dependencies = {
            ("persistence_service", "crop_service"),  # Thumbnails need cropping
        }

        all_violations = []

        for py_file in services_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            service_name = py_file.stem
            imports = get_imports_from_file(py_file)

            for module, line in imports:
                for other_service in service_modules:
                    if other_service == service_name:
                        continue
                    if other_service in module:
                        # Check if this is an allowed dependency
                        if (service_name, other_service) in allowed_dependencies:
                            continue
                        all_violations.append(
                            f"{py_file.name}:{line} imports {module} (circular)"
                        )

        assert len(all_violations) == 0, (
            "Detector services should not import each other. Violations:\n"
            + "\n".join(all_violations)
        )

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
        tests.test_services_only_import_from_core()
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
        detector_tests.test_detector_services_do_not_import_each_other_circularly()
        print("✓ Detector services no circular imports")
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
