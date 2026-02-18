# WatchMyBirds Agent Governance

## Authority Model
- `docs/INVARIANTS.md` is the architecture source of truth.
- Rules marked as `HARD` are mandatory and blocking.

## Cross-Impact Completion Rule (HARD)
- If a change affects cross-cutting contracts or operations (for example ports, firewall, service names, environment variables, API shape, storage paths, or build/deploy behavior), required linked updates must be completed in the same task.
- If immediate completion is not possible, add a delegation entry to `docs/IMPACT_LEDGER.md` before merge.
- Do not mark a task complete while required cross-impact updates are missing and undocumented.

## CI and Review Policy
- PR checks must fail on `HARD` rule violations.
- Cross-impact checks are enforced by workflow gates in `.github/workflows/`.

## Language
- All user-facing app text must be English.
- All repository documentation must be English.
