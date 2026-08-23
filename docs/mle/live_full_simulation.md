# Live full-simulation MLE

`live-simulation` controls one private shared-runtime acquisition causally. The
estimator receives a truth-free handshake, fits only durable observations, selects
the next reachable detector station and shield program, and finally binds its output
to the immutable MeasurementLog published by the runtime.

The runtime owns the physical experiment profile. This includes the environment,
candidate isotopes, physical transport configuration, station and measurement limits,
views per station, live time, minimum station separation, and coverage radius. The MLE
repository owns only its estimator, planning objective, convergence rule, diagnostics,
and posterior reporting.

## Configuration ownership

The standard live estimator files are:

- `configs/mle/live_surface_spectral.json` for MLE numerical settings;
- `configs/mle/live_surface_planning.json` for Fisher-information planning settings;
- `configs/mle/live_surface_stop.json` for compound statistical stopping gates.

The spectral config intentionally omits `isotope_names`. The planning config
intentionally omits `live_time_s` and `shield_program_length`. Those values are bound
from the runtime experiment profile after the adaptive handshake. Supplying any of
these duplicated fields is an error.

The standard runtime profile currently supplies a 10 x 15 x 5 m environment, 16
stations, 8 views per station, 20 s per view, 128 measurements, 3 m minimum station
separation, and 3 m coverage radius. These values are documented here for operators;
they are not implemented or defaulted by the estimator.

## Preflight

Preflight verifies that the shared runtime checkout, Geant4 sidecar, response registry,
full-fidelity transport settings, and local MLE configs are compatible without starting
an acquisition:

```bash
uv run estimate-radiation-mle live-simulation --preflight-only --json
```

The runtime must use external Geant4 with full unit-weight transport, detector-directed
`detector_cps_1m` source-rate semantics, incident-energy detector scoring, the configured
detector response, and line-resolved shield attenuation. Thinning, weighted histories,
analytic transport, and theory-TVL attenuation are rejected.

## Start a live acquisition

Author the private scenario in the shared runtime repository. Scene realization and
truth manifests stay private to that repository; the estimator is passed only the
opaque scenario path required to start the session:

```bash
uv run estimate-radiation-mle live-simulation \
  --scenario /private/live-scenario.json \
  --output-dir results/live-mle \
  --json
```

For a long Geant4 acquisition, launch the command in a persistent `tmux` session and
redirect output to a timestamped log. When the CUI dashboard starts, the command prints
its browser URL immediately.

## Causal and stopping contract

The controller sends exactly one runtime request at a time. A multi-view shield program
uses a single station ID, marks intermediate views incomplete, and marks only the final
view complete. MLE fitting and station planning occur only at durable station boundaries.

The estimator may stop early only when all configured convergence, coverage,
identifiability, residual, and expected-information gates pass. The runtime-owned
maximum measurement count is the hard safety limit; the CLI provides no independent
override.

On completion, the runtime publishes the immutable MeasurementLog. The estimator
validates the log against the truth-free experiment profile, binds the final report to
the log digest, and publishes its result outside the MeasurementLog directory.

## Resume

A staged adaptive stream can be resumed after its last verified complete station:

```bash
uv run estimate-radiation-mle live-simulation \
  --scenario /private/live-scenario.json \
  --resume-stage /runtime/.measurement-log.stream-17 \
  --resume-compatibility /runtime/resume-compatibility.json \
  --output-dir results/live-mle
```

The runtime validates scenario and provenance compatibility. The estimator reconstructs
its causal state from the verified durable prefix before selecting another station.
