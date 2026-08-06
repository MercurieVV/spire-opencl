# spire-opencl

Instantiate numeric code at a symbolic value type, get an OpenCL kernel.

```scala
libraryDependencies += "io.github.mercurievv" %% "spire-opencl" % "0.1.0-SNAPSHOT"
```

## The idea

Code written against `Field`, `ConvertableTo` and friends does not care what `V` is. Instantiate it at
`Double` and it computes a number; instantiate it at `Expr` and it *builds the tree it would have
computed*. That tree is an IR, and an IR can be compiled.

```scala
import io.github.mercurievv.spireopencl.symbolic.{Expr, Reify, instances}
import io.github.mercurievv.spireopencl.opencl.ClKernel
import instances.given

// polymorphic, knows nothing about this library:
def ramp[V: Field](x: V, slope: V): V = x * slope

val formula = Reify(uniforms = List("dx"), params = List("slope")) { (uniform, param) =>
  ramp(Expr.mul(Expr.Index, uniform("dx")), param("slope"))
}

ClKernel.compile[IO](formula, size = 128, maxBatchSize = 4).use { kernel =>
  IO(kernel.renderBatchUnsafe(
    uniforms = Map("dx" -> 0.01f),
    batch    = Seq(Map("slope" -> 1.0f), Map("slope" -> 2.0f)),
    out      = new Array[Float](128 * 2)))
}
```

Reification is by **application**, not inspection: the function is applied to symbolic inputs and the
result is kept. No macros, no free-arrow encoding.

## The three ways a value enters a kernel

| | varies | becomes |
|---|---|---|
| `Expr.Const` | never | an inlined literal, constant-folded where possible |
| `Expr.Uniform(name)` | per launch | a scalar kernel argument |
| `Expr.Param(name)` | per batch element | a slot in one packed, element-major buffer |
| `Expr.Index` | per work-item, dimension 0 | `get_global_id(0)`, as a float |

Arguments are **declared, not discovered**. Constant folding can delete a name from the tree, and a
derived argument list would silently shift kernel argument indices under it.

Anything varying *within* a launch is derived from `Expr.Index` by the caller, in the IR. The library
does not know what dimension 0 counts.

## Reduction

`Expr.Sum(body)` sums over the batch dimension. It is not a post-processing step: the kernel keeps the
batch dimension parallel and reduces through work-group local memory, so the sum happens while the
elements are still in registers. Everything above the `Sum` in the tree runs once, in the reducing
work-item — so a scale factor applied above it costs one multiply per work-item rather than one per
element.

Valid only at the root of a formula.

## Measured behaviour

From the audio synthesizer this was extracted from (M3 Max, OpenCL 1.2, 128 work-items):

- **Per-launch latency is the cost at small sizes**, not the arithmetic: 260 µs at 1 batch element,
  258 µs at 16. Batch, do not loop.
- **An idle GPU costs ~750 µs to wake.** Launch p50 by idle gap: 258 µs (back-to-back), 590 µs (500 µs
  gap), 1050 µs (2600 µs gap) — and flat in batch size at every gap. Back-to-back benchmarks overstate
  what a duty-cycled caller gets.
- **One packed parameter buffer, written non-blocking.** A `clEnqueueWriteBuffer` per parameter doubled
  launch cost (271 → 552 µs).
- **Do not serialise the batch dimension.** Collapsing it into a `for` loop inside each work-item cost
  3128 µs against 1012 µs at 16 elements: 128 work-items alone cannot fill the device.

## Limits

- **Single precision.** Apple Silicon has no `cl_khr_fp64`; a `double` kernel does not build there.
  `DeviceProbe` reports what a given machine actually supports.
- **Operations:** `+ - * /`, `sin`, and `>`/`<` as 0/1 masks. No loops, no branches, no per-element
  state.
- **No `spire.algebra.Trig[Expr]`** — it requires `exp`, `log`, `atan` and the rest. `TrigonometryCC`
  (this library's one-function typeclass) is provided instead; a stub that threw would let code compile
  and fail at reification rather than at the call.
- `Expr.Sum` may not nest — there is one batch dimension.

## Check your machine

```
mill spireOpencl.runMain io.github.mercurievv.spireopencl.opencl.DeviceProbe
```

Prints every platform and device, the `cl_khr_fp64` verdict, and whether a float and a double kernel
actually build and run — the extension string and the preferred-width hint both lie on some drivers.

## Releasing

The git tag is the version — `publishVersion` reads `GITHUB_REF_NAME`, so:

```
git tag v0.1.0 && git push --tags
```

CI tests on every push and publishes to Maven Central on a `v*` tag. Four repository secrets are
required (Settings → Secrets and variables → Actions):

| secret | what it is |
|---|---|
| `SONATYPE_USERNAME` / `SONATYPE_PASSWORD` | a **user token** from [central.sonatype.com](https://central.sonatype.com) → Account → Generate User Token. Not the portal login. |
| `PGP_SECRET_BASE64` | `gpg --export-secret-keys --armor <keyid> \| base64` — the signing key, which must also be published to a keyserver |
| `PGP_PASSPHRASE` | that key's passphrase, if it has one |

The `io.github.mercurievv` namespace has to be verified once on the portal (it proves ownership via
the GitHub account). Locally, `./mill spireOpencl.publishLocal` needs none of this.

## Licence

Apache 2.0.
