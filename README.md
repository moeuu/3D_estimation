# Radiation Surface MLE Estimator

This repository owns only the 3-D surface maximum-likelihood estimator. Physical
acquisition, Geant4, detector/shield/obstacle physics, spectrum generation, and raw
MeasurementLog publication are provided by the
`rotating-shield-simulation-runtime` dependency.

Sharing the runtime does not mean sharing estimator settings. MLE keeps its own
surface dictionary, optimizer, Fisher planner, measurement budget, and compound
stopping rule; PF keeps separate particles, SMC/RJ kernels, EIG planner, budgets,
and stopping criteria. Only the physical workspace, action-execution protocol, and
raw observation contract are common.

## Online estimator

`OnlineMLESession` is the runtime-facing API. A controller passes each
`runtime.records.MeasurementRecord` only after the shared runtime has durably staged
it. The MLE buffers every shield-view record without fitting an incomplete station.
When the final view marks `station_complete`, it performs exactly one coarse
all-history warm fit for that measurement point and publishes the durable report.
Live shutdown uses three explicit phases. `complete_live_state()` rebuilds and seals
the configured full-resolution map while acquisition is still active;
`bind_finalized_measurement_log()` then authenticates the immutable runtime log;
`publish_bound_result()` only adds provenance and writes the final artifacts. No
scientific fit is allowed after runtime log publication, and `OnlineMLESession` has
no ambiguous `finalize()` method.
The output is:

```text
OUTPUT_DIR/
├── online_state.json
├── index.html
├── dashboard_data.json
├── stations/
│   └── station_000000_step_00000000/
│       ├── mle_estimate.npz
│       ├── mle_diagnostics.json
│       └── hotspot_clusters.json
└── final/
    ├── mle_estimate.npz
    ├── mle_diagnostics.json
    └── hotspot_clusters.json
```

The class never generates observations and never writes a MeasurementLog. The runtime
controller retains this ordering:

```python
observation_session.step(action)  # runtime simulates and durably stages first
record = observation_session.writer.records[-1]
online_mle.receive_persisted(
    record,
    station_complete=action.station_complete,
)
```

The live `ral-full-simulation` launcher starts the dashboard server by default and
immediately prints a
clickable `CUI split visualization URL: http://HOST:PORT/index.html` before the first station
fit. If the requested port belongs to an older run, the launcher selects the next
free port instead of displaying stale files. The page refreshes from atomically
published MLE snapshots;
it neither reads nor displays simulation truth. Use `--dashboard-port`,
`--dashboard-host`, and `--dashboard-public-host` for remote delivery. Set
`MLE_DASHBOARD_PUBLIC_HOST` when the externally reachable host should be discovered
from the environment. `--no-serve` keeps the static dashboard files without opening
a server, while `--no-dashboard` disables both.

Production acquisition remains a shared-runtime operation:

```bash
uv run --project ../Rotating-shield-simulation-runtime \
  rotating-shield-sim run-adaptive-session /private/physical-scenario.json
```

## MLE information planning

After a completed station, the MLE can rank runtime-supplied detector positions and
jointly choose an eight-measurement Fe/Pb orientation program:

```python
planned = online_mle.plan_next_action(
    candidate_poses_xyz=runtime_candidates_xyz,
    travel_costs=runtime_travel_costs,
    planning_config=MLEPlanningConfig.load(
        "configs/mle/default_planning.json"
    ),
)
next_action = planned.selected_action
```

This is MLE-specific local Fisher `D_s`-optimal design, not the PF particle EIG rule.
Background and scatter are marginalized by a Schur determinant. The complete shield
program is selected jointly as one station block, while also marginalizing one
shared future station-rate nuisance. Thus complementary relative shield signatures,
not merely independently strong views, determine the ordered program. Zero MLE
regions remain represented by residual surface exploration modes. Direct floor/ceiling
response separation, alternative-support separation, z-Fisher information, response
correlation reduction, elevation diversity, and geometry exploration supplement the
local determinant. Candidate generation, collision filtering, local spatial
refinement, routes, and observation production stay in the shared runtime. The
selected action includes detector xyz plus an ordered
`measurement_program` with Fe/Pb indices, dwell times, and the final
`station_complete` flag.

The online dashboard served at its existing URL shows the latest recommendation and
marks its detector position. See [MLE information planning](docs/mle/information_planning.md)
for the criterion, live candidate contract, and limitations.

## RA-L full simulation

The physical RA-L profile is owned by the shared runtime; this repository provides a
strict launcher for runtime acquisition under live spectral MLE control. Verify the
complete Geant4/model/config chain without starting a long run:

```bash
uv run estimate-radiation-mle ral-full-simulation \
  --preflight-only --json
```

Run a private physical scenario under live MLE control and publish the dashboard:

```bash
uv run estimate-radiation-mle ral-full-simulation \
  --scenario .local/private-scenarios/ral-mix9-scenario.json \
  --output-dir results/ral-runs/ral-mix9-mle \
  --json
```

The private scenario is created explicitly for each run; it is not searched for or
generated by this repository. Its exact runtime schema and the mapping from a
PF-style MIX-9 source layout are documented in
[RA-L full simulation](docs/mle/ral_full_simulation.md#create-the-private-scenario-for-each-run).
For the Cs-137 x4, Co-60 x3, Eu-154 x0 absent-isotope case, generate runtime profile
`ral-cs4-co3-eu0` and pass `--private-scene-profile ral-cs4-co3-eu0` when starting
the live launcher.

The launcher accepts only a private live scenario and uses
`rotating-shield-sim run-adaptive-session`. The online coarse MLE fits once after
the current shield program closes its station;
intermediate spectra are buffered without changing the estimate. It can then ask the
runtime to refine promising 3-D candidate neighborhoods, rerank the returned
reachable poses, and select the next position and eight-measurement Fe/Pb program.
No station count, view count, record count,
measurement route, or station list is fixed in advance. Source truth is never passed
to the MLE estimator, planner, or estimator-owned CUI process. The live CUI is
truth-free; evaluation overlays must be produced by a separate post-estimation
evaluator after the estimator result is final.

The RA-L stop decision is compound: convergence/KKT, independent 3-D pose and height
coverage, elevation span, held-history deviance/map/cluster stability, response
correlation, floor/ceiling separation, systematic residuals, and a patience window of
low expected information gain must all pass. `--max-measurements` remains only a
safety bound.

See [RA-L full-simulation launcher](docs/mle/ral_full_simulation.md) for ownership,
persistent `tmux` execution, recovery of the same adaptive session, and the
adaptive-controller boundary.

## Repository layout

```text
configs/mle/             versioned estimator configurations
docs/mle/                architecture and runtime data-contract notes
examples/legacy/         archived IAS notebook, data, and figures
fixtures/                small contract/conformance fixtures
scripts/                 maintenance and conformance entry points only
src/three_d_estimation/  online adapter, response assembly, solver, reporting
tests/mle/                estimator and runtime-boundary tests
```

Generated `build/`, `dist/`, `results/`, `images/`, caches, and root-level script
wrappers are intentionally absent. Use the installed `estimate-radiation-mle` entry
point rather than `main.py` or `mle_main.py`.

The IAS-era research workflow is preserved under `examples/legacy/`; its complete
pre-runtime source snapshot is also retained by the Git tag
`legacy-ias-mle-before-pf-runtime-import`. It is not imported by production code.

## Development

Python 3.12 and `uv` are required.

```bash
uv sync
uv run python scripts/check_repository_boundary.py
uv run pytest
```

Released under the [MIT License](LICENSE).
