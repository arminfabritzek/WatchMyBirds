# WatchMyBirds Contributing Guide

Thank you for your interest in contributing to WatchMyBirds! 🐦

## Getting Started

### Local Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
The web UI starts on `http://localhost:8050`.

### Docker
```bash
cp docker-compose.example.yml docker-compose.yml
docker-compose up -d
```

## Coding Style

- **Python:** Follow PEP 8, 4-space indentation.
- **Formatter:** [Black](https://github.com/psf/black) (`line-length = 88`).
- **Linter:** [Ruff](https://docs.astral.sh/ruff/). Rules are in `pyproject.toml`.
- **Type Hints:** Python 3.11+ type hints are required for new code.

Run the formatting and linting suite:
```bash
ruff check --fix .
black .
```

## Testing

- Tests live in `tests/`.
- Run with `pytest`.
- Please add tests when your change touches core logic.

## Commit & Pull Request Guidelines

- Use clear, scoped commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
  `fix: handle missing network interface`, `feat: add species filter`, `docs: update setup instructions`.
- PRs should include a short summary and list the affected areas.

## Architecture Notes

- Architectural constraints and invariants are documented in `docs/`.
- Please review them before proposing structural refactors.

## Questions?

Open an issue or start a discussion — we're happy to help!
