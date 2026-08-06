# spire-opencl

Instantiate numeric code at a symbolic value type, get an OpenCL kernel.

```scala
libraryDependencies += "io.github.mercurievv" %% "spire-opencl" % "0.1.0-SNAPSHOT"
```

## The idea

Write the program in spire. Say what its arguments are. Run it on the GPU.

```scala
import spire.algebra.{Field, Trig}
import spire.implicits.*
import spire.math.sin

// Ordinary spire. Nothing here imports this library or knows a GPU exists.
def program[V: {Field, Trig}](b: V, c: V, d: V): V = b * c - sin(d)
```

```scala
import io.github.mercurievv.spireopencl.symbolic.{Reify, instances}
import io.github.mercurievv.spireopencl.opencl.ClKernel
import instances.given

val formula = Reify(uniforms = Nil, params = List("b", "c", "d")) { (_, arg) =>
  program(arg("b"), arg("c"), arg("d"))     // V = Expr, so this builds the tree
}

ClKernel.compile[IO](formula, size = 1, maxBatchSize = 1).use { kernel =>
  IO {
    val out = new Array[Float](1)
    kernel.renderUnsafe(Map.empty, Map("b" -> 2.5f, "c" -> 4.0f, "d" -> 0.75f), out)
    out(0)                                  // 9.318..., same as program[Double](2.5, 4.0, 0.75)
  }
}
```

`program` is called exactly the same way in both instantiations. At `V = Double` it computes a number;
at `V = Expr` the arithmetic *is* the tree, because `Field[Expr]` and `Trig[Expr]` build nodes instead
of folding numbers. That tree is the IR, and an IR can be compiled.

Reification is by **application**, not inspection: the function is applied to symbolic inputs and the
result is kept. No macros, no free-arrow encoding, no DSL to learn.

The interesting case is when the program has structure worth compiling — a thousand-term sum, a big
polynomial, one kernel reused across a batch:

```scala
val formula = Reify(uniforms = List("dx"), params = List("slope")) { (uniform, param) =>
  ramp(Expr.mul(Expr.Index, uniform("dx")), param("slope"))
}

kernel.renderBatchUnsafe(
  uniforms = Map("dx" -> 0.01f),
  batch    = Seq(Map("slope" -> 1.0f), Map("slope" -> 2.0f)),
  out      = new Array[Float](128 * 2))    // both slopes, one launch
```

## What spire gives you at `Expr`

| typeclass | what it buys |
|---|---|
| `algebra.ring.Field` | `+ - * /`, `zero`, `one` — so `b * c - d` is just spire syntax |
| `spire.algebra.Trig` | `sin cos tan asin acos atan atan2 sinh cosh tanh exp expm1 log log1p`, `e`, `pi`, `toRadians/Degrees` |
| `spire.algebra.NRoot` | `sqrt`, `nroot`, `fpow` — an integer exponent unrolls to multiplication rather than calling `pow` |
| `spire.math.ConvertableTo` | numeric literals of any type enter the tree |
| `OrderS` (this library) | `>` / `<` as **0/1 masks**, not booleans — branchless gates that multiply into an expression |

Every one of those maps to an OpenCL built-in with the same semantics as `java.lang.Math`, and each
operation carries its own `Double` implementation in the IR, so constant folding, the reference
interpreter and the kernel cannot disagree about what `Div` means.

Constants fold as you build: `program[Expr](Const(2), Const(3), Const(0))` *is* `Const(6.0)`.

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
  `DeviceProbe` reports what a given machine actually supports. The IR itself holds `Double`; only the
  emitted kernel narrows.
- **Straight-line code only.** No loops, no data-dependent branches, no per-element state. A comparison
  is a mask you multiply by, which covers conditionals but not iteration.
- **No integer or boolean type.** Everything is a float, including the 0/1 masks.
- `Expr.Sum` may not nest — there is one batch dimension, and it reduces once at the root.
- Not provided: `Ring`-only or exact-arithmetic instances (`Rational`, `Algebraic`), ordering that
  returns `Boolean`, `Bits`/integral typeclasses.

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
