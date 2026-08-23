# Surface MLE architecture

This document describes the implemented all-history surface maximum-likelihood estimator. For commands and setup, start with the [repository README](../../README.md). For the persisted acquisition contract, see [measurement-log format](measurement_log.md).

## Repository and estimator boundary

Production acquisition is not owned by this repository. The versioned
`rotating-shield-simulation-runtime` package produces and durably stages truth-free raw
MeasurementLog schema-2 records. This estimator imports that package's public record,
observation-model, continuous-kernel, asset-resolution, and final-log validation APIs.
It does not copy or synchronize runtime source.

Production has one live estimator path. `OnlineMLESession` accepts each
already-persisted runtime record. Production live mode buffers all shield views at
one point and performs one coarse all-history warm fit only when the durable station
marker closes that measurement point. `complete_live_state()` rebuilds and seals the
configured full-resolution grid and uncertainty result before runtime publication.
Only then may the controller publish the immutable MeasurementLog, bind its exact
context and record digest with `bind_finalized_measurement_log()`, and serialize the
already-sealed estimate with `publish_bound_result()`. The last two phases cannot fit
or otherwise mutate scientific state. After each completed station,
`OnlineMLESession.plan_next_action` ranks the current truth-free runtime candidate
poses and Fe/Pb programs using local Fisher `D_s`-optimal design, explicit
vertical/support ambiguity criteria, and the shared physical kernel. No command
reloads a saved log and estimate to plan another action.

No path reads PF state or PF candidates. Runtime candidate generation, route and
traversability planning, acquisition execution, simulation, and MeasurementLog
writing remain outside this package. This package owns only MLE-based candidate
ranking and shield-program selection. Cross-repository forward-response conformance
tests detect estimator/kernel drift. The planning derivation and action contract are
documented in [MLE information planning](information_planning.md).

The live integration follows the same boundary. `live-simulation` performs a
strict physical/configuration preflight and passes a private scenario path directly
to `rotating-shield-sim run-adaptive-session`. The runtime publishes reachable
truth-free candidates and each durably staged spectrum. The MLE refits all causal
history, asks the runtime for local candidate refinement when useful, and selects the
next runtime-sized station program. No action list or route appears in the scenario;
station, view, live-time, and measurement limits come from the runtime experiment
profile. The immutable log is validated and bound to
the final MLE report only after the compound
stability/coverage/ambiguity/information stop rule fires. The enforced order is
scientific completion, runtime log publication, exact binding, then artifact
publication. See the [live closed-loop runbook](live_full_simulation.md).

## Data contracts and dimensions

The notation used below is:

| Symbol | Meaning |
| --- | --- |
| `M` | durably staged measurements in the current causal prefix |
| `B` | spectrum energy bins |
| `G` | active surface patches |
| `I` | isotope channels, in manifest/config order |
| `Q` | quadrature points per patch, currently 1 or 4 |
| `E` | physical shared-edge graph edges |
| `N` | fitted nuisance coefficients |

The online backend converts the records already delivered by the runtime into one
validated `ObservationBatch` through a pathless runtime array view:

| Field | Shape | Meaning |
| --- | --- | --- |
| `detector_positions_xyz` | `M × 3` | detector centers in metres |
| `detector_quaternions_wxyz` | `M × 4` | detector orientations |
| `fe_indices`, `pb_indices` | `M` | selected Fe/Pb shield orientations |
| `live_times_s` | `M` | finalized dwell per observation |
| `spectrum_counts` | `M × B` | raw, non-negative spectral counts |
| `energy_bin_edges_keV` | `B + 1` | common strictly increasing bin edges |
| `isotope_counts` | `M × I`, optional | processed `response_poisson` channels |
| `isotope_covariances` | `M × I × I`, optional | preserved extraction covariance |

All records must use exactly the same energy edges. Count and covariance fields must
either be available consistently for all records or be absent. Count-domain fitting
requires `isotope_counts`; spectral live fitting uses `spectrum_counts`. Count mode supports
the historical Poisson diagnostic and covariance-aware Gaussian or multivariate
Student-t fitting. Covariance matrices are regularized, condition-checked, and used
jointly across isotope channels. Spectral MLE remains the authoritative result because
count covariance cannot correct systematic spectrum-decomposition bias.

## Physical units

The logged source-rate model is exactly `detector_cps_1m`. The MLE unknown is a surface density:

```text
rho[g,i]                         detector cps@1m/m²
area[g]                          m²
s[g,i] = area[g] * rho[g,i]     detector cps@1m
```

`cps@1m` is detector count-rate-equivalent strength at one metre. It is not source activity in Bq or an emitted-photon rate. Keeping density and integrated strength distinct prevents patch resolution from silently changing physical meaning.

The saved estimate exposes both organizations:

- `density_by_isotope`: `I × G`, in detector `cps@1m/m²`;
- `patch_strength_by_isotope`: `I × G`, in detector `cps@1m`.

## Surface discretization

`build_surface_patches` constructs axis-aligned rectangular elements for:

- uncovered room floor;
- full room ceiling;
- all four room walls;
- obstacle tops;
- obstacle sides exposed to free space.

Blocked floor cells are split on exact obstacle boundaries and removed. Faces between adjacent blocked cells are not emitted, and an obstacle side coincident with a room boundary is skipped. Room normals point inward; obstacle normals point out of the solid and into free space.

`patch_spacing_m = [sx, sy, sz]` is a target spacing, not an assumed exact cell size. Each face axis uses `ceil(length / spacing)` intervals so the final vertices remain on the exact room or obstacle boundary. Every patch validates that:

- its four vertices form an oriented rectangle;
- its stored area equals the vertex cross-product area;
- its centroid and unit normal agree with the geometry;
- quadrature points lie inside the rectangle;
- quadrature weights are positive and sum to one.

`quadrature_order: 1` uses the centroid with weight 1. `quadrature_order: 4` uses a two-dimensional 2 × 2 Gauss rule with four equal weights. The weights approximate the normalized average over a patch; area is never hidden in them.

Patches have stable IDs, optional parent IDs, and refinement levels. Solver adjacency uses dense active-patch indices, while reports also retain stable-ID adjacency. Two patches are neighbors only when they share a positive-length physical edge. The edge weight is that length in metres, including perpendicular room-face joins and coarse/fine overlaps after selective refinement.

## Shared physical kernel

The online backend builds the shared runtime's `RuntimeObservationModel` and
`ContinuousKernel` directly from the live `RunContext`. Their response includes
detector/source geometry, finite detector and aperture settings, selected Fe/Pb shield
geometry and attenuation, obstacle path attenuation, optional buildup, and calibrated
transport-response terms. File-backed assets are resolved by the runtime package from
the MeasurementLog root or its versioned package assets.

For measurement `m`, patch `g`, isotope `i`, and normalized quadrature point `r`, the integrated-strength count response is:

```text
R_int[m,g,i] = live_time[m] * sum_r(weight[g,r] * K_i(m, point[g,r]))
R_density[m,g,i] = area[g] * R_int[m,g,i]
```

Thus `R_int` maps integrated patch `cps@1m` to expected counts, while `R_density` maps surface `cps@1m/m²` to expected counts.

### Count-domain response

`build_count_responses` first constructs `M × G × I` integrated-strength and density responses. The estimator embeds the integrated-strength response diagonally into `M × I × G × I`: a processed isotope count channel is driven by the matching isotope's surface map.

With count channel `c`, the mean is:

```text
mu[m,c] = sum_g sum_i 1[c == i] * R_int[m,g,i] * s[g,i]
          + live_time[m] * background_rate[c]
```

The background basis is optional and has one non-negative rate coefficient per isotope channel.

### Line-resolved spectral response

`build_spectral_response` can construct the full diagnostic tensor:

```text
M × B × G × I
```

integrated-strength response tensor. Each isotope's positive gamma-line weights are normalized. Each line is evaluated with its own Fe/Pb attenuation coefficients and, when obstacle transport attenuation is present, its own obstacle attenuation row. Spectral fitting requires those line-resolved tables rather than silently substituting an aggregate attenuation value.

For each line, the spatial kernel is combined with a CeBr3 detector response across energy bins, including efficiency, energy resolution, Compton continuum, and configured backscatter fraction. Contributions from all lines are summed into the isotope axis. Multiplying by `area[g]` gives the corresponding density response.

The expected spectrum is:

```text
mu[m,b] = sum_g sum_i R_spec[m,b,g,i] * s[g,i]
          + sum_n nuisance_basis[m,b,n] * nuisance[n]
```

The optional spectral nuisance bases are non-negative, live-time-scaled, normalized background and scatter shapes. They fit one background rate and one scatter rate, rather than one unconstrained value per bin.

Production live fitting instead creates a `ResponseOperator`. Its forward and adjoint
products stream measurement, energy-bin, patch, quadrature, and isotope blocks without
materializing the full tensor. Per-measurement physical blocks use content-addressed
disk caching, so an extended causal prefix evaluates only new rows. The cache key binds
the physical model, runtime manifest, detector/shield state, energy edges, patch
geometry, and response settings.

When a runtime discrepancy-calibration artifact is configured, spectral nuisance
columns additionally cover calibrated background/scatter shapes, low-dimensional
all-64-pair shield leakage, station/pose shared rate, signed low-rank residual modes,
and optional gain/resolution derivatives. Family-specific L2 shrinkage prevents these
columns from freely absorbing source response. The calibrated negative-binomial
dispersion models remaining overdispersion. A real independent artifact is mandatory;
the estimator never fabricates a run-specific correction.

## Objective and solver

Both modes flatten their observation dimensions and solve the same non-negative convex problem. With density `rho`, nuisance coefficients `u`, expected counts `mu`, patch area `a_g`, and shared-edge length `ell_gh`, the implemented objective is:

```text
sum_j (mu[j] - y[j] * log(mu[j]))
+ lambda_l1    * sum_g sum_i a[g] * rho[g,i]
+ lambda_tv    * sum_(g,h) ell[g,h] * sum_i abs(rho[g,i] - rho[h,i])
+ lambda_group * sum_g a[g] * norm2(rho[g,:])
+ lambda_n1    * sum_n u[n]
+ 0.5 * lambda_n2 * sum_n u[n]^2
```

All density and nuisance variables are constrained non-negative, and `min_mean` protects the Poisson log at zero.

The terms have distinct roles:

- Area-weighted L1 penalizes total integrated source strength and promotes sparse support without making the cost depend arbitrarily on patch size.
- Edge-length-weighted TV promotes piecewise-smooth density over physically connected surfaces.
- The isotope group penalty applies an L2 norm across isotopes at each patch, then sums those group norms with area weights. It promotes common spatial support while retaining isotope-specific amplitudes.
- Nuisance L1/L2 terms control non-negative count background or spectral background/scatter rates.

`fit_surface_map_poisson` uses a diagonally preconditioned Chambolle-Pock primal-dual
iteration with a closed-form likelihood-conjugate proximal update, TV dual clipping,
non-negative L1 steps, exact per-patch group shrinkage, and optional over-relaxation.
The streamed solver supports Poisson and calibrated negative-binomial likelihoods on
CPU or CUDA. It checks both relative state change and relative objective change at
`check_interval`, and stops when both requested tolerances are met or
`max_iterations` is reached.

The covariance-aware count diagnostic uses constrained L-BFGS-B over the same
non-negative density/nuisance variables with the full per-measurement isotope
covariance. Multivariate Student-t is the robust default for this secondary mode.

## Coarse-to-fine and debias stages

When `coarse_to_fine_levels > 0`, each level:

1. ranks active positive patches by total integrated strength across isotopes;
2. selects `ceil(refinement_fraction * active_positive_count)` strongest patches;
3. replaces each selected rectangle with four area-preserving children;
4. rebuilds complete physical adjacency, including coarse/fine shared edges;
5. transfers the parent's density, not total strength, to each child as a warm start;
6. resolves the full all-history problem on the refined grid.

Stable child IDs are allocated above the current maximum, and every child retains its parent ID and incremented refinement level.

When `debias_refit` is enabled, the final regularized solution selects patch/isotope support at:

```text
density[g,i] >= support_threshold_fraction * max_g(density[g,i])
```

Response columns outside that support are zeroed. The selected problem is then refit with structural L1, TV, and group weights set to zero. Nuisance penalties remain configured. This reduces shrinkage bias without reopening the entire support.

## Fit/hold-out split and diagnostics

`held_out_fraction` deterministically selects complete groups using `random_seed`; at least one measurement remains in the fit. `held_out_grouping` defaults to `station_id`, so related shield views at one station cannot cross the fit/held-out boundary. `same_xy_height` additionally keeps vertically separated records at the same XY location together, while `shield_program_block` groups the explicit block IDs preserved from observation metadata. Only the explicit diagnostic `row` mode permits individual rows to split. The estimator fits only the training groups, predicts every row, and reports held-out Poisson deviance when the held-out set is non-empty.

Every estimate records:

- mode, units, patch count, and response tensor shape;
- fit and held-out measurement indices;
- held-out grouping, per-row group labels, and selected group IDs;
- full residual arrays and residual L2 norm;
- Poisson negative log likelihood and total deviance;
- L1, TV, group, and nuisance penalty contributions;
- objective history, iteration count, convergence flag, relative state/objective changes, and normalized KKT residual;
- fitted nuisance names;
- covariance-likelihood family, regularization, and per-row condition numbers in
  count mode;
- line energies and normalized line weights in spectral mode;
- response matrix rank, nonzero-singular-value condition number, zero columns, and highly cosine-correlated source-column pairs;
- connected hotspot clusters;
- active-support Laplace covariance, station-bootstrap patch selection frequencies,
  cluster centroid/strength intervals, surface-mass probabilities, z intervals, and
  ceiling probability when final uncertainty is enabled.

Hotspot extraction thresholds each isotope relative to its own peak, forms connected components on the physical shared-edge graph, optionally filters by integrated `cps@1m`, and reports stable patch IDs, strength-weighted centroid, integrated strength, peak density, and involved surface kinds.

## Regularization selection and uncertainty

`regularization_selection: grouped_cv` evaluates the configured L1/TV grid using
whole station or same-XY-height groups. It selects the strongest regularization within
one standard error of the minimum validation deviance when requested. Evaluation on
an unseen environment is a separate post-estimation concern and is not an estimator
command that refits a completed log.

Final uncertainty is explicitly conditional on the selected support. The active
response columns form a regularized Fisher/Laplace covariance; a hard parameter cap
prevents accidental dense inversion. Station-block bootstrap estimates selection
frequency and refits complete station groups. Reports aggregate these samples into
patch selection frequencies, cluster centroid covariance, centroid and integrated
strength intervals, isotope/surface mass probabilities, z intervals, and ceiling
probability. This is not a claim of globally exact post-selection uncertainty; the
selection frequencies and response-correlation diagnostics expose instability.

## Reporting

`save_mle_estimate` stages deterministic, pickle-free output before publishing it:

- `mle_estimate.npz` contains numerical arrays, patch geometry, quadrature, stable adjacency plus ordered patch metadata from which dense solver adjacency is reconstructed, refinement lineage, density/strength maps, predictions, nuisance coefficients, and summary scalars.
- `mle_diagnostics.json` contains diagnostics and the fully resolved MLE configuration.
- `hotspot_clusters.json` mirrors cluster diagnostics when present.

NPZ member order and ZIP timestamps are fixed. The NPZ includes the SHA-256 of the exact diagnostics JSON. Loading verifies that binding, schema versions, mirrored summary values, all array shapes, patch-level neighbor symmetry, and global adjacency. Existing report members are not overwritten unless `--overwrite` is supplied.

## CPU, GPU, cache, and memory

`--cpu` forces the NumPy streamed path. The production live profile requests CUDA and
fails closed if the configured device is unavailable; smaller default profiles remain
CPU-friendly.

`--gpu` runs shared physical-kernel evaluation and optimization through PyTorch
tensors on `gpu_device` (default `cuda`). The request is strict: missing
PyTorch/device support raises an error rather than falling back. `gpu_dtype` accepts
`float32` or `float64`; the production live profile retains `float64`.

`response_measurement_chunk_size`, `response_energy_chunk_size`, and
`response_patch_chunk_size` bound streamed spectral blocks. The default measurement
chunk is eight, so one complete station enters each shared-runtime kernel call
together. Gamma-line detector pulses are constructed once per line and reused across
all measurement/patch chunks. `response_cache_dir` stores one contiguous full-energy
file per content-addressed measurement/patch chunk; emitted energy blocks are sliced
from that file in memory. This keeps causal prefix reuse while preventing the cache
inode count and load calls from scaling with the number of energy chunks. Cache and
prefix-reuse diagnostics report both physical files and emitted blocks. CPU cache
misses are distributed over a bounded process
pool controlled by `response_worker_count`; zero selects the hardware-aware default,
while CUDA kernel construction remains single-launch-streamed to avoid competing
allocations. During a CUDA solve, the exact float64 response is opportunistically
cached as one device matrix when it fits within
`response_device_cache_fraction` of currently free VRAM. This removes all iterative
host-to-device response transfers. If it does not fit, the solver preserves the same
objective and normally falls back to bounded streamed blocks. Production profiles
may set `require_gpu_response_cache=true`; those fits fail before iteration with the
required, free, and budgeted byte counts instead of silently accepting an extreme
streamed runtime. Construction time, solver time, cache bytes, response-product
calls, and estimated iterative transfer bytes are included in response-operator
diagnostics. Materialized builder mode remains available only for small deterministic
diagnostics and equivalence tests.

The device cache persists across causal station fits. A station extension copies the
resident prefix device-to-device and transfers only the new eight rows; bootstrap
station permutations and duplicate rows are gathered from the same resident matrix
without rebuilding physics or transferring response blocks from the host. Row sums,
column sums, and device-cache population share one block traversal. Final bootstrap
replicates are scheduled in exact concurrent batches controlled by
`bootstrap_batch_size`; each replicate retains its original station resample,
configured bootstrap solver dtype, convergence checks, refinement, and debias path.
Live estimation reports phase changes, checked iterations, KKT residuals,
maximum-iteration ETA, and completed bootstrap replicates on stderr. JSON summaries
therefore remain clean on stdout.

Online and final spatial budgets are separate: `online_patch_spacing_m` and
`online_coarse_to_fine_levels` define low-latency causal fits, while
`patch_spacing_m` and `coarse_to_fine_levels` define the final report. A coarse warm
state is never silently applied to an incompatible final topology.

## Configuration map

The implemented public JSON fields are grouped below:

| Concern | Fields |
| --- | --- |
| Mode/channels | `mode`, `isotope_names` |
| Surface grid | `patch_spacing_m`, `quadrature_order`, `obstacle_height_m` |
| Structural penalties | `l1_weight`, `tv_weight`, `isotope_group_weight` |
| Nuisance model | background/scatter and calibrated leakage/station/low-rank/drift switches, `discrepancy_calibration_path`, nuisance penalties |
| Likelihood | `spectral_likelihood`, `count_likelihood`, Student-t and covariance-conditioning fields |
| Regularization selection | `regularization_selection`, grouped CV fields, tuning/final environment IDs |
| Iteration | `max_iterations`, `poisson_em_warm_start_iterations`, `tolerance`, `objective_tolerance`, `kkt_tolerance`, `check_interval`, `step_safety`, `over_relaxation`, `min_mean` |
| Compute | response mode/chunk/cache fields, `require_gpu_response_cache`, `use_gpu`, `gpu_device`, `gpu_dtype` |
| Online/final split | `online_fit_scope=station_complete`, `online_patch_spacing_m`, `online_coarse_to_fine_levels` |
| Spectral pulse | `continuum_to_peak`, `backscatter_fraction` |
| Refinement/debias | `coarse_to_fine_levels`, `refinement_fraction`, `debias_refit`, `support_threshold_fraction` |
| Uncertainty | Laplace support/ridge/cap and station-bootstrap fields |
| Post-processing | `response_correlation_threshold`, `cluster_threshold_fraction`, `cluster_min_strength_cps_1m` |
| Evaluation split | `held_out_fraction`, `held_out_grouping`, `held_out_xy_tolerance_m`, `random_seed` |

Examples are checked in under [configs/mle](../../configs/mle). The CLI validates these fields through `MLEConfig`; unknown fields or physically invalid values fail rather than being ignored.
