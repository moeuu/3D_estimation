# AGENTS.md

## Repository role

This repository owns only the 3-D surface-MLE estimator. Geant4, environment,
detector, shield, obstacle transport, spectrum physics, observation generation,
and raw MeasurementLog ownership belong to the sibling
`Rotating-shield-simulation-runtime` package.

## Repository isolation

- Depend on the pinned `rotating-shield-simulation-runtime` package rather
  than copying its implementation.
- Do not vendor `native`, `measurement`, `sim`, `spectrum`, environment assets,
  source layouts, obstacle layouts, or MeasurementLog writers here.
- Keep MLE solver, regularization, response-matrix assembly, warm starts, and
  MLE reporting under `src/three_d_estimation/`.
- Do not add finalized-MeasurementLog all-history fitting commands or offline
  estimator runners. Production estimation advances through observations
  delivered durably to `OnlineMLESession`.
- Simulation acquisition is invoked through the shared runtime CLI, never by a
  local simulator implementation.

## Git publishing

- Commit and push this repository only on the `main` branch.
- Never create or push `agent/*`, feature, release, or any other branch.
- Before every push, verify that the current branch is exactly `main` and use
  the explicit command `git push origin main`.
- Do not push tags or open pull requests unless the user explicitly requests
  that separate action.

## Physical semantics

- `intensity_cps_1m` is expected net detector cps at 1 m, not total gamma/s.
- MLE code must call the shared `RuntimeObservationModel` and
  `ContinuousKernel`; do not duplicate shield, obstacle, detector, or spectrum
  physics in the estimator.
- Preserve spectra, bin edges, variances, detector poses, live times, and Fe/Pb
  orientation indices in measurement logs.
- Production observations must be delivered by the shared runtime, not from
  expected-count substitutions.

## Performance

- Batch response construction across measurements, patches, quadrature points,
  shield pairs, and spectrum bins.
- A scalar implementation is allowed only as a deterministic test oracle.
- Every GPU path needs a CPU/GPU equivalence test.

## Style and tests

- Use Python 3.12 and `uv`.
- Follow PEP 8; every function must have a docstring.
- Comments and docstrings are written in English.
- Run `uv run pytest` before completion.
- Run `uv run python scripts/check_repository_boundary.py` before completion.
