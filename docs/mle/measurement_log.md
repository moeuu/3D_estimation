# Shared MeasurementLog boundary

`MeasurementLog` is the estimator-independent boundary between acquisition and the
surface MLE. The versioned `rotating-shield-simulation-runtime` package exclusively
owns its schema, durable stream writer, validation, and final publication. This
repository only consumes the package's `MeasurementLog`, `RunContext`, and
`MeasurementRecord` views.

The log contains resolved physical model metadata and raw unit-weight integer spectra.
It never contains realized source truth, PF particles/candidates, MLE output, or
planner state.

## Published directory

```text
RUN_DIR/
├── run_manifest.json
├── runtime_config.resolved.json
├── environment.json
├── forward_model_manifest.json
├── observations.npz
├── observation_metadata.jsonl
└── repository_commit.txt
```

The runtime loader verifies the file inventory and artifact hashes before exposing any
record. Truth and source-layout artifacts must live outside `RUN_DIR`; forbidden truth
names and recursively embedded realized source values fail closed.

## Raw observation arrays

For `M` records and `B` energy bins, `observations.npz` contains:

| Member | Dtype | Shape |
| --- | --- | --- |
| `step_id`, `action_id`, `station_id` | `int64` | `M` |
| `detector_pose_xyz` | `float64` | `M × 3` |
| `detector_quat_wxyz` | `float64` | `M × 4` |
| `fe_orientation_index`, `pb_orientation_index` | `int64` | `M` |
| `live_time_s`, `travel_time_s`, `shield_actuation_time_s` | `float64` | `M` |
| `energy_bin_edges_keV` | `float64` | `B + 1` |
| `spectrum_counts` | `int64` | `M × B` |

Spectra are exact non-negative event counts. Schema v2 deliberately has no projected
isotope counts, fitted spectrum variances, or isotope covariance arrays. Production
MLE therefore uses the spectral online backend. The count-domain
solver remains available only for an explicitly derived/imported count contract; it
must not project raw v2 spectra itself.

Live time is positive. Travel and shield-actuation times are non-negative. Fe/Pb
orientation indices lie in `[0, 7]`. Quaternions are normalized, energy edges are
strictly increasing, steps/actions follow zero-based causal order, and station IDs are
contiguous nondecreasing groups.

## Station-causal online use

The runtime stream writer first fsyncs a record shard and its metadata row. On the
final record at a station it then durably rewrites that row with
`station_complete: true`. Only after those operations may a controller call
`OnlineMLESession.receive_persisted`.

The MLE does not read private stream-stage files. It receives the public
`MeasurementRecord` object held by the runtime controller, and it validates any
explicit station flag against the durable record metadata. Non-final station views are
buffered without solving. At a station boundary the backend fits the complete prefix,
using the prior station solution only as numerical initialization.

Each live station report records:

- covered step IDs;
- cutoff step and station;
- covered-record content SHA-256;
- update policy `station_complete_all_history`; and
- runtime/config identity without claiming a full-log hash that was unavailable at
  that causal cutoff.

The final report may bind the complete finalized MeasurementLog content hash.

## Forward-model identity

`forward_model_manifest.json` binds the MLE response to the runtime's physical
semantics. The current native contract includes:

- repository and resolved runtime-config identity;
- detector, shield, environment, obstacle, transport, and spectrum model IDs/hashes;
- detector-response, shield-pose, obstacle-material, and transport-table contract
  hashes;
- `detector_cps_1m` source-rate semantics;
- distance/time/energy/attenuation units;
- all 8 × 8 Fe/Pb orientation pairs;
- line-resolved isotope energies, weights, and Fe/Pb attenuation; and
- obstacle path and live-time response semantics.

The online backend and neutral prefix-scoring/planning context reconstruct
`RuntimeObservationModel` and `ContinuousKernel` from this validated runtime context.
They do not maintain a second copy of detector, shield, obstacle, or spectrum physics.

## Source-rate semantics

`intensity_cps_1m` and fitted integrated patch strength both mean expected net detector
count rate at one metre for the configured detector and processing contract. They are
not total gamma emission rate or Bq. Surface density has units
`detector_cps_1m / m²`.

## Publication and content digest

During acquisition, the runtime's `MeasurementLogStreamWriter` persists deterministic
per-record shards before estimator ingestion. Finalization consolidates them into the
canonical pickle-free bundle, verifies one station marker per station, fsyncs the
artifacts, and atomically publishes the output directory without replacing an existing
log.

`MeasurementLog.content_sha256` inventories every regular non-truth artifact below the
published directory and hashes that inventory. MLE final reports retain this digest;
online station reports instead bind only their causally covered records.
