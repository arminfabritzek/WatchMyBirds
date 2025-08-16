# Codex Project Instructions - WatchMyBirds Inference App

## 1. Code Guidelines
- Use the guidelines from AGENTS.md.


## 2. Interaction with Codex
- Codex must:
  - Always consider this file and `AGENTS.md`.
  - Only modify production scripts, never files in `legacy/`.
  - Automatically update the documentation according to Section 8 after **every** code change.
- Add TODO comments if uncertain.

---

## 7. Dependencies
- Python 3.11+
- Dependencies in `requirements.txt`.
- Update and document the file for new packages.

---

## 8. Documentation Maintenance Policy
- **Documentation must be updated after every code change.**

---


### Mandatory Updates
1. **README.md** and **PROJECT_STATUS.md**
   - Synchronize folder structure, config keys, and instructions.  
   - Add test and usage notes if changed.  

### Rules
- Documentation changes must be made **in the same commit** as code changes.
- Do not leave outdated information in these files (update).
- If unclear → completely regenerate the file with the current state.

---

### Additional Notes
- `README.md` and `PROJECT_STATUS.md` must always reflect the actual state of the pipeline.
- Documentation maintenance is part of every development task.