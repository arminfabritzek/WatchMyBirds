# WatchMyBirds Contributing Guide

## Project Structure & Module Organization
- Core Python app lives at the repo root (`main.py`, `web/`, `detectors/`, `camera/`, `utils/`, `models/`).
- Web UI templates are in `templates/` and static assets in `assets/`.
- Raspberry Pi appliance and first‑boot/AP logic are in `rpi/`.
- Analytics dashboard is a separate Svelte/Vite app in `analytics/`.
- Runtime output is written under `output/` (images, DB, settings).

## Build, Test, and Development Commands
- Local Python run:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python main.py
  ```
  Starts the Flask app on `http://localhost:8050`.
- Docker quickstart:
  ```bash
  cp docker-compose.example.yml docker-compose.yml
  docker-compose up -d
  ```
  Runs the full stack with mounted `output/` and `models/`.
- Analytics (optional UI module):
  ```bash
  cd analytics
  npm install
  npm run dev   # local dev server
  npm run build # production build
  ```

## Coding Style & Tooling
- **Python:** Follow PEP 8, 4-space indentation. Keep modules focused and avoid side-effects.
- **Formatter:** **Black** is the mandatory formatter (`line-length = 88`).
- **Linter:** **Ruff** is the mandatory linter. Rules are managed in `pyproject.toml`.
- **Type Hints:** Python 3.11+ type hints are mandatory for new functions and classes.
- **Path handling:** **never** build storage paths manually; use `utils/path_manager.PathManager`.
- **Image processing:** shared logic must live in `utils/image_ops.py`.
- **UI:** new features must use Flask/Jinja2 (Dash is deprecated).

### Linting & Formatting Workflow
You can run the standard linting and formatting suite using the agent workflow:
```bash
/lint
```
Alternatively, run manually:
```bash
ruff check --fix .
black .
```

## Testing Guidelines
- The test suite is located in `tests/`.
- Run tests using `pytest`.
- For changes touching storage or deletion logic, add focused tests to protect invariants (immutability of `originals/`, disposable `derivatives/`).
- Use `pyproject.toml` to configure pytest options.

## Commit & Pull Request Guidelines
- Git history shows **no consistent commit convention** (mixed short messages like “asdf” and descriptive ones).
- Prefer clear, scoped commit messages (e.g., `fix: handle missing wlan0` or `docs: clarify first-boot AP flow`).
- PRs should include a concise summary, affected areas (e.g., `rpi/`, `web/`, `detectors/`), and any operational impacts.

## Security & Configuration Tips
- Change the default UI password (`EDIT_PASSWORD`) after first login.
- RPi first‑boot AP mode only runs once; ensure `rpi/first-boot/first-boot.sh` and `rpi/systemd/wmb-first-boot.service` are in the image.
- Keep `output/` backed up; it contains `images.db` and originals.

## Architecture Constraints (Read Before Refactors)
- Invariants are documented in `ARCHITECTURE.md` and `INVARIANTS.md`.
- Originals are immutable; derivatives are disposable; DB is the metadata authority.
- Storage changes must update `utils/path_manager.PathManager` and review deletion logic (`utils/file_gc.py`).
