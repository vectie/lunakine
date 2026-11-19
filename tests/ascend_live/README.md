# Live Ascend actor comparison

Build this diagnostic with the same native runtime as the deployment CLI:

```sh
moon build tests/ascend_live --release --target native --deny-warn
./_build/native/release/build/tests/ascend_live/ascend_live.exe LIBASCENDCL ACTOR.om ACTOR.json MOONROBO_COMPARISON.json NEW_REPORT.json
```

Use an actual AscendCL library, a model compiled for the observed SoC, and the
MoonRobo comparison artifact containing saved histories, goals, LibTorch actions
and portable-reference actions. The executable reuses one session and its
input/output buffers for1,000 calls, rotating over every fixture case. It checks
every action against both saved outputs at absolute tolerance0.001 and rejects
nonfinite differences. Ten calls are excluded from latency statistics.

Timing covers the synchronous `Session::run_into` call, including host/device
transfers, and excludes fixture conversion, output comparison and JSON writing.
The report records minimum, mean, p95, p99 and maximum microseconds. Both the
session and runtime close before the report is written. This is a sampled actor
comparison and short reuse measurement; it does not qualify robot timing,
telemetry calibration, physical control or indefinite sustained use.

CPU/native test doubles cannot establish hardware acceptance. Retain the host's
driver/card identity and ATC precision settings alongside any reported result.
This probe adds no dependencies or branches to the production actor runtime.
