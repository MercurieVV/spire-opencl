# spire-opencl

Instantiate numeric code at a symbolic value type, get an OpenCL kernel.

Every fence on these pages is checked by `mill docs.mdoc` against the real library on every run — none
of this can drift from the actual API.

- [README](README.md) — the core idea: reify a spire program at `V = Expr`, compile it, run it.
- [Var operations](var-operations.md) — more `Field`/`Trig`/`NRoot` ops at once, plus a `Var`/`.at`-built
  stateful cell, compiled and actually launched on a real OpenCL device.

Source: [github.com/MercurieVV/spire-opencl](https://github.com/MercurieVV/spire-opencl).
