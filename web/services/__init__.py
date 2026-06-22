"""
WatchMyBirds Services Package.

This package contains service layer classes that encapsulate business logic,
separating it from Flask routes for better testability and maintainability.

ARCHITECTURE RULE (HARD invariant H-01):
- Services may import from core/*, stdlib/typing, config, logging_config,
  utils.*, and other web.services.* modules
- Services MUST NOT import directly from camera/* or detectors/*

Import services explicitly by submodule (``from web.services import
gallery_service`` / ``from web.services.foo import bar``). This package
intentionally does NOT eagerly import its submodules: a barrel import here
coupled every service's transitive dependencies (e.g. ONVIF discovery
pulling ``camera.network_scanner`` and ``ifaddr``) to merely touching any
one unrelated service, which made narrow tests fragile against optional
deps.
"""
