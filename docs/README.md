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
libraryDependencies += "io.github.mercurievv" %% "spire-opencl" % "0.1.4"
```

Mill:

```scala
def mvnDeps = Seq(
  mvn"io.github.mercurievv::spire-opencl:0.1.4"
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

// Plain Spire code: no OpenCL imports and no library-specific DSL.
def program[V: {Field, Trig}](b: V, c: V, d: V): V =
  b * c - sin(d)

// With V = Double it is just a normal CPU calculation.
program[Double](2.5, 4.0, 0.75)
```

## Build a formula

Run the same function with symbolic arguments:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Reify, instances}
import instances.given

val formula = Reify(uniforms = Nil, params = List("b", "c", "d")) { (_, param) =>
  program(param("b"), param("c"), param("d")) // V = Expr, so operators build an expression tree.
}
```

```scala mdoc
formula.params
```

`Reify` does not inspect Scala code. It applies your function to `Expr` values and keeps the expression
that comes back. Declared names matter: uniforms are scalar launch arguments, params vary by batch
element, and inputs are device-resident arrays.

## Inspect the kernel

`CodeGen` lowers the formula to OpenCL C:

```scala mdoc
import io.github.mercurievv.spireopencl.opencl.CodeGen

CodeGen(formula).linesIterator.take(6).mkString("\n")
```

## Compile and launch

The launch path needs real OpenCL hardware, so this fence is compile-only in the documentation build.
It still type-checks against the real API on every docs run.

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.ClKernel

val result: Float =
  ClKernel.compile[IO](formula, size = 1, maxBatchSize = 1).use { kernel =>
    IO {
      val out = new Array[Float](1)
      kernel.renderUnsafe(
        uniforms = Map.empty,
        params = Map("b" -> 2.5f, "c" -> 4.0f, "d" -> 0.75f),
        out = out,
      )
      out(0)
    }
  }.unsafeRunSync()
```

Expected output is about `9.318361`, matching the `Double` example up to Float precision.

## Arrays

Use `Reify.arrays` when each work-item reads one element from a device-resident array:

```scala mdoc:silent
def elementwise[V: Field](a: V, b: V, c: V): V =
  a * b + c

val arrayFormula = Reify.arrays(
  uniforms = Nil,
  params = Nil,
  inputs = List("a", "b", "c"),
) { (_, _, input) =>
  elementwise(input("a"), input("b"), input("c"))
}
```

```scala mdoc
CodeGen(arrayFormula).linesIterator.find(_.contains("__global const float* a")).get
```

Upload inputs separately from launches:

```scala mdoc:compile-only
import cats.effect.IO
import io.github.mercurievv.spireopencl.opencl.ClKernel

val n = 1024
ClKernel.compile[IO](arrayFormula, size = n, maxBatchSize = 1).use { kernel =>
  IO {
    kernel.writeInputUnsafe("a", Array.fill(n)(1.5f))
    kernel.writeInputUnsafe("b", Array.fill(n)(2.0f))
    kernel.writeInputUnsafe("c", Array.fill(n)(0.25f))

    val out = new Array[Float](n)
    kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
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
