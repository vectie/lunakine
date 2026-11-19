# LunaKine implementation guide

LunaKine is a MoonBit-native actor inference runtime. MoonRobo owns training,
robot observations, motion references, and SDK/control integration. LunaFlux
owns LLM inference. Do not import either product into this runtime.

Use the MoonBit agent guide and native C-binding guide for relevant work.
Keep portable actor contracts, offline conversion, and device execution in
separate packages. Public concrete types belong to public packages. Production
Ascend execution uses CANN/AscendCL through a narrow private native boundary;
it does not load Python, PyTorch, or CUDA. Reference evaluation is an explicit
offline correctness tool, never a silent device fallback.

Prefer simple shape/numerical checks and deterministic native resource release.
Do not add hashing, artifact certification, generic admission frameworks, or
runtime compiler layers without a concrete execution requirement. Use `.mbtx`
for automation. Run affected native checks/tests, then `moon info` and
`moon fmt`; review generated interfaces. Hardware claims require real hardware
results, and tests using a native test double must be labeled accordingly.
