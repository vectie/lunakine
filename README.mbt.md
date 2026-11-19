# LunaKine

LunaKine is a native actor inference runtime, initially targeting compiled
models on Huawei Ascend. It is a separate repository from LunaFlux (LLM
inference) and MoonRobo (LibTorch/CUDA training and robot control).

The public Ascend API exposes Runtime/Session ownership and numeric run-into
calls. Native handle types remain private; callers receive no vendor handles.

The runtime executes numeric actor programs. Robot calibration, motion
references, observations, and SDK commands stay with its caller. Portable
artifacts and offline conversion are separate from native device execution.

Packages are deliberately separate:

| Package | Purpose |
| --- | --- |
| `actor` | Portable ELU encoder/decoder specification, FP32 parameter encoding, and explicit offline reference evaluation. |
| `onnx` | Static batch-one ONNX export with normalization, ELU/tanh, branch ordering, and action masking. No training framework or Python dependency. |
| `ascend` | Explicit AscendCL runtime/session ownership and preallocated FP32 input/output buffers for compiled models. |
| `cmd/export_onnx` | Convert a portable bundle to ONNX. |
| `cmd/run` | Execute a compiled actor through a selected AscendCL library. |

The portable bundle contains `actor.json` and `parameters.f32`. MoonRobo's
export also writes `actor.onnx` and its own `robot.json` companion. Parameter
storage is little-endian FP32: each encoder layer's row-major `[out,in]` weights
then bias, followed by each decoder layer in the same order. Numeric history is
oldest-to-newest; reference preview is a second input. Robot joint interpretation
belongs to MoonRobo, not this runtime.

From the LunaKine root:

```sh
moon check --target native --deny-warn
moon run scripts/test_ascend_abi.mbtx /absolute/path/lunakine
moon run cmd/export_onnx --target native -- /absolute/bundle /absolute/actor.onnx
moon run scripts/compile_actor.mbtx /absolute/bundle SOC_VERSION /absolute/compiled_actor PRECISION_MODE --dry-run
moon run cmd/run --target native -- /absolute/libascendcl.so /absolute/compiled_actor.om /absolute/bundle/actor.json /absolute/input.json
```

The run command accepts flattened `history` and `goals` arrays in `input.json`.
Create one runtime per process and share it across sessions for one selected
device, with a single caller at a time. Close sessions before the runtime.
Successful native runs reuse device buffers and datasets; model I/O must be
static FP32 with input names `history`, `goals` and one action output.

ATC compilation requires an installed compatible CANN toolchain and the exact
reported SoC. Precision is explicit; the compiler script does not choose a card
or precision mode automatically. Internal FP16 execution with FP32 I/O still
requires numerical qualification. Missing libraries or unsupported model I/O
produce errors rather than falling back to reference execution.

The software ABI tests load a small, explicitly selected test library and check
transport, error propagation, buffer reuse, and resource balance. They do not
simulate NPU arithmetic. `scripts/test_ascend_asan.mbtx` adds an ASan smoke check
without changing the shared MoonBit toolchain; on macOS mimalloc remains enabled
and leak checking is not provided by that smoke run.

An existing full-size E1 actor has been exported and compared with LibTorch and
independent ONNX Runtime on three fixed input sets. On2026-11-17, its actual
Ascend310P3 deployment passed CANN9.1.0 ATC compilation with FP32 I/O and
`must_keep_origin_dtype`, then1,000 native NPU calls checking29,000 actions.
After the authorized reboot on kernel6.8.0-149, the same comparison passed
again: maximum absolute LibTorch error2.861e-6 and transfer-inclusive mean/p99
latency2.333/2.392ms. Driver26.1.1 loads automatically and firmware9.0.0.9.220
is confirmed active. The separate `tests/ascend_live` diagnostic records this
sampled comparison. Sustained timing and physical robot acceptance remain
pending. See MoonRobo's
[phase plan](../moonrobo/docs/LUNAKINE_IMPLEMENTATION_PLAN.md) for exact evidence
and remaining implementation work.
