# Impact Delegation Ledger

Use this file when a cross-impact change cannot be fully completed in one task.

If all required linked updates are completed immediately, no entry is needed.

## Entry Template

```md
## YYYY-MM-DD - Short Title
- Change summary:
- Why not fully completed now:
- Required follow-up changes:
  - [ ] Item 1
  - [ ] Item 2
- Affected files/modules:
- Owner:
- Target date:
- Tracking reference (issue/PR/task):
```

## Open Entries

## 2026-02-18 - Wave 1 Backport Cross-Impact Delegations
- Change summary: Wave 1 backport applied runtime, DB, docker, and deployment hardening changes while intentionally deferring coupled follow-ups.
- Why not fully completed now: Wave 1 excluded backend-coupled review features by design and focused on low-risk additive scope. A boundary refactor for service layering (H-01) is larger than Wave 1 risk budget.
- Required follow-up changes:
  - [ ] Create `docs/streaming.md` with bridge-networking and stream source mode behavior.
  - [ ] Wire `web/services/analysis_service.py` into an explicit runtime startup path in Wave 2/3.
  - [ ] Resolve H-01 service-boundary violations in `web/services/analysis_service.py` by moving non-core dependencies behind a compliant interface (`cv2`, `config`, `web.services`, `web.services.weather_service`).
  - [x] Replace local camera IP placeholder in `docker-compose.yml` with non-local generic value.
- Affected files/modules: `docker-compose.yml`, `docker-compose.example.yml`, `web/services/analysis_service.py`, `docs/streaming.md`, `docs/INVARIANTS.md`
- Owner: Maintainer team
- Target date: Wave 2/Wave 3 execution window
- Tracking reference (issue/PR/task): Public Wave 1 backport execution
