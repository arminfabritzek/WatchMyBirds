# AGENTS.md – Guidelines for Codex and Agents

> **Purpose**: Ensure that all agents (including Codex) consistently maintain code quality and documentation

---

1. Installation Instructions
- Install missing Python packages:  
  ```bash
  pip install -r requirements.txt

	•	Python-Version: 3.11+
	•	For new dependencies: update requirements.txt and document them.

⸻

2. Code Style Rules

Sprache	Formatter / Linter	Besondere Regeln
Python	Black (88 Zeichen), UTF-8	Do not use from … import *
JavaScript (ES6)	ESLint (airbnb-base)	Use files only inside static/

2.1 Naming Conventions
	•	snake_case for variables and functions
	•	PascalCase for classes
	•	Boolean variables: is_…, has_…, should_…
	•	Loop indices i/j only in very short loops (< 5 lines)
	•	No cryptic abbreviations (df → pandas_dataframe)

2.2 Docstrings
	•	Triple-quoted """ directly below the definition.
	•	First line: verb in 3rd person singular (German).
	•	Multiline for complex functions:

Args:
    name (type): Description.
Returns:
    type: Description.
Raises:
    ErrorType: When the error occurs.

	•	Example:

def lade_daten(pfad: str) -> pd.DataFrame:
    """Lädt Daten aus einer CSV-Datei."""

⸻

3. Commit Messages

	•	Format: <type>: <short summary>
	•	Types: feat, fix, refactor, docs, style, test, build, ci
	•	Example:

feat: add hyperparameter tuning with Optuna

⸻

4. Security Guidelines
	•	Only modify production scripts; do not change files in legacy/.
	•	Tests (pytest) must pass before merging changes.

⸻

5. Documentation
	•	Every code change must update documentation in the same commit:
	•	PROJECT_STATUS
	•	Add new configuration parameters if applicable.


⸻

6. Consistency with CODEX.md
	•	All agents must, before execution:
	•	Read CODEX.md (project rules)
	•	Read AGENTS.md (work guidelines)
	•	Instructions from both files have the highest priority.

⸻

7. CI/CD and Automation
	•	GitHub Actions or other automation must only run if:
	•	Documentation files are up to date.
	•	Tests are successful.
	•	If tests fail → do not automatically merge changes.

⸻

🔑 Important:
	•	Maintaining documentation is a fixed part of every development task.
	•	Codex must not finalize changes without these updates.
	•	Before every commit, read **PROJECT_STATUS.md** and **CODEX.md** to know the current phase and objectives.
	•	Changes to the codebase must implement these objectives.

---

