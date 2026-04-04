# Changelog

## 0.1.1 - 2026-04-04

This patch release consolidates the April 4, 2026 improvements into a single
versioned release so GitHub releases and Docker tags can point at the same
stable build.

### Highlights

- Upgrade Raspberry Pi build and runtime paths to Python 3.12.
- Harden the Raspberry Pi setup flow by requiring an admin password and
  tightening the first-boot experience.
- Improve the review workflow with refined quick species selection states and a
  larger review species artwork set.
- Refresh setup and deployment documentation for the Python 3.12 baseline.
- Improve the live stream overlay so the clock stays responsive and shows only
  temperature plus humidity.

### Included 2026-04-04 commits

- `855a1c1` `build(rpi): migrate build and runtime to python 3.12`
- `20b9c91` `security(rpi): require admin password and harden setup flow`
- `49d773d` `docs: refresh python 3.12 and setup guidance`
- `67665c5` `feat(review): refine quick species selection states`
- `9dab33d` `assets: expand review species artwork`
- `e9d3191` `Make stream clock overlay responsive and show temp + humidity only`
