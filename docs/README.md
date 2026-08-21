{%
laika.title = Start
laika.metadata.canonicalLink = "https://mercurievv.github.io/spire-opencl/"
%}

# spire-opencl

`spire-opencl` is a Scala 3 library for GPU numeric computing with Typelevel Spire and OpenCL. It
turns ordinary Spire typeclass code into an OpenCL kernel by running that code once with symbolic
values.

These pages are checked by `./mill docs.mdoc`: `scala mdoc` fences compile, and non-hardware examples
also run. Generate the HTML site with `./mill docs.site`; mdoc runs first, then Laika renders the
checked markdown.

## Setup

SBT:

```scala
libraryDependencies += "io.github.mercurievv" %% "spire-opencl" % "@VERSION@"
```

Mill:

```scala
def mvnDeps = Seq(
  mvn"io.github.mercurievv::spire-opencl:@VERSION@"
)
```

You also need an OpenCL 1.2 runtime and driver. Check what the library sees:

```bash
./mill spireOpencl.runMain io.github.mercurievv.spireopencl.opencl.DeviceProbe
```

The generated kernels use `Float`. The IR stores constants as `Double`, but `CodeGen` emits single
precision OpenCL because common devices, including Apple Silicon, do not expose `cl_khr_fp64`.

## Write one numeric program

Write the computation once, polymorphic in the value type:

```scala mdoc
import spire.algebra.{Field, Trig}
import spire.implicits.*
import spire.math.sin

// Plain Spire code: no OpenCL imports and no library-specific DSL. Args groups the arguments, Answer
// names the result — both ordinary case classes, nothing OpenCL-specific about either.
case class Args[V](b: V, c: V, d: V)
case class Answer[V](value: V)

def program[V: {Field, Trig}](args: Args[V]): Answer[V] =
  Answer(args.b * args.c - sin(args.d))

// With V = Double it is just a normal CPU calculation.
program(Args[Double](2.5, 4.0, 0.75))
```

## Build a formula

Run the same function with symbolic arguments — `Reify.outTyped[Args, Answer]` reads `Args`'s field
labels for the params and `Answer`'s for the result name, both off the case classes themselves:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Reify, TypedFormula, TypedFormula2, instances}
import instances.given

val formula: TypedFormula2[Args, Answer] = Reify.outTyped[Args, Answer](uniforms = Nil) { (_, args) =>
  program(args) // V = Expr, so operators build an expression tree.
}
```

```scala mdoc
formula.formula.params
formula.outputNames
```

`Reify` does not inspect Scala code. It applies your function to `Expr` values and keeps the expression
that comes back. Declared names matter: uniforms are scalar launch arguments, params vary by batch
element, and inputs are device-resident arrays.

## Inspect the kernel

`CodeGen` lowers the formula to OpenCL C:

```scala mdoc
import io.github.mercurievv.spireopencl.opencl.CodeGen

CodeGen(formula.formula).linesIterator.take(6).mkString("\n")
```

## Compile and launch

The launch path needs real OpenCL hardware, so this fence is compile-only in the documentation build.
It still type-checks against the real API on every docs run. `ClKernel.compileT2` + `TypedKernel2.renderT`
take an `Args[Float]` and hand back an `Answer[Float]` — no `Map`, no `out` array:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderT}

val result: Answer[Float] =
  ClKernel.compileT2[IO, Args, Answer](formula, size = 1, maxBatchSize = 1).use { kernel =>
    IO { kernel.renderT(Args[Float](b = 2.5f, c = 4.0f, d = 0.75f)) }
  }.unsafeRunSync()
```

Expected `result.value` is about `9.318361`, matching the `Double` example up to Float precision.

## Arrays

Use `Reify.arraysTyped` when each work-item reads one element from a device-resident array — one input
per field, matched by name, the array analogue of `outTyped`:

```scala mdoc:silent
case class Vals[V](a: V, b: V, c: V)

def elementwise[V: Field](vals: Vals[V]): V =
  vals.a * vals.b + vals.c

val arrayFormula: TypedFormula[Vals] = Reify.arraysTyped[Vals](uniforms = Nil, params = Nil) { (_, _, vals) =>
  elementwise(vals)
}
```

```scala mdoc
CodeGen(arrayFormula.formula).linesIterator.find(_.contains("__global const float* a")).get
```

Upload inputs separately from launches — `TypedKernel.writeInputsT` takes a `Vals[Array[Float]]`, one
array per field, instead of one `writeInputUnsafe` call per array:

```scala mdoc:compile-only
import cats.effect.IO
import io.github.mercurievv.spireopencl.opencl.{ClKernel, writeInputsT}

val n = 1024
ClKernel.compileT[IO, Vals](arrayFormula, size = n, maxBatchSize = 1).use { kernel =>
  IO {
    kernel.writeInputsT(Vals[Array[Float]](
      a = Array.fill(n)(1.5f),
      b = Array.fill(n)(2.0f),
      c = Array.fill(n)(0.25f),
    ))

    val out = new Array[Float](n)
    kernel.kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
    out(0) // 3.25f
  }
}
```

Inputs stay on the device until written again. This is the right shape for repeated kernels over data
that changes less often than the launch parameters.

## Value kinds

| kind | varies | use |
|---|---|---|
| `Expr.Const` | never | literals and folded constants |
| `Expr.Uniform(name)` | per launch | scalars shared by the whole launch |
| `Expr.Param(name)` | per batch element | small per-case values packed for one launch |
| `Expr.Input(name)` | per work-item | large arrays resident on the device |
| `Expr.Index` | per work-item | the OpenCL global id as a float |

Supported numeric surface: `Field`, `Trig`, `NRoot`, numeric literals through `ConvertableTo`, and
`OrderS` comparisons as `0`/`1` masks. There are no booleans, integers, loops, or data-dependent
branches in the generated kernel.

More examples:

- [State and more operations](var-operations.md)
- [Benchmarking](benchmarking.md)
