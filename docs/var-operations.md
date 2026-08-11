# Var-driven operations, run on a real device

A second walkthrough of the idea in [README](README.md): this one leans on more of `Field`/`Trig`/`NRoot`
at once, and carries state across launches with `Var`/`.at` instead of hand-writing an update `Expr`. Every
fence below is checked by `mill docs.mdoc`, and the kernel fence actually launches on this machine's OpenCL
device — not `compile-only`.

## A program with a few different operations

Ordinary spire, generic over `V`. `+`, `-`, `*`, `/`, `sqrt` and `sin` all come from typeclasses, so nothing
here is aware a GPU exists:

```scala mdoc
import spire.algebra.{Field, NRoot, Trig}
import spire.implicits.*
import spire.math.{sin, sqrt}

def program[V: {Field, NRoot, Trig}](x: V, y: V, z: V): V =
  val ratio = (x - y) / (z + 1)
  sqrt(ratio * ratio) + sin(y)

program[Double](3.0, 1.0, 4.0)
```

## A stateful cell, built from `Var`

`Var[F, V, A]` is one cell's step: previous value in, next value and an output out. `.at` places it in a
`Store` keyed by id — the zoom from "a step that knows one value" into "a program that threads many". Here
the cell is an exponential moving average: it reads its own previous output, blends in the new sample, and
hands back what it held *before* this update (so launch `n`'s output reflects launches `0..n-1`, matching
how the state buffer behaves on device):

```scala mdoc
import cats.Id
import cats.data.StateT
import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify, instances}
import io.github.mercurievv.spireopencl.symbolic.state.{Store, Var, at}
import instances.given

val alpha = Expr.Uniform("alpha")

val smooth: Var[Id, Expr, Expr] =
  StateT { prev =>
    val sample = Expr.Param("x")
    val next = prev + (sample - prev) * alpha
    (next, prev)
  }

val formula: Formula =
  Reify.statefulVar(uniforms = List("alpha"), params = List("x")) { (_, _) =>
    smooth.at[Store[Expr]](0)
  }

formula.states
```

`Reify.statefulVar` runs `smooth.at(0)` once from an empty store to discover which cell ids the program
touches, then again seeded with an `Expr.State` read for each — so the cell's slot comes from the program's
own shape, not from inspecting the folded tree.

## Running it on the device

`ClKernel.compile` builds and links real OpenCL C from `formula`; the resource below is used and torn down
within this fence, on whatever device `ClKernel.defaultDevice` finds (a GPU if one is present):

```scala mdoc
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.ClKernel

val emaOnDevice: Vector[Float] =
  ClKernel.compile[IO](formula, size = 1).use { kernel =>
    IO {
      val samples = Vector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f)
      val out = new Array[Float](1)
      samples.map { x =>
        kernel.renderUnsafe(Map("alpha" -> 0.5f), Map("x" -> x), out)
        out(0)
      }
    }
  }.unsafeRunSync()

emaOnDevice
```

Same recurrence, computed on the host for comparison — `0`, then each step blending the previous value
half-way toward `1.0`:

```scala mdoc
val emaOnHost: Vector[Float] =
  Vector.iterate(0.0f, 5)(prev => prev + (1.0f - prev) * 0.5f)

emaOnHost
```

`emaOnDevice == emaOnHost`: the cell that flowed through `Var`/`.at`/`Reify.statefulVar` is not simulated —
it is the same sequence the compiled kernel produced, read back off the device after each launch.
