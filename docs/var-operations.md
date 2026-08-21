# State and More Operations

Two features beyond the first kernel: a wider Spire surface and a small stateful cell. mdoc-checked.

## More Spire operations

Grouping related arguments into a case class beats a run of same-typed positional ones — `richProgram(x, y, z)` doesn't tell
call sites which is which, `richProgram(point)` does:

```scala mdoc
import spire.algebra.{Field, NRoot, Trig}
import spire.implicits.*
import spire.math.{sin, sqrt}

case class Point[V](x: V, y: V, z: V)

def richProgram[V: {Field, NRoot, Trig}](point: Point[V]): V =
  val ratio = (point.x - point.y) / (point.z + 1)
  sqrt(ratio * ratio) + sin(point.y)

richProgram(Point[Double](3.0, 1.0, 4.0))
```

Reifying it still needs one `Expr` per field, but `Reify[Point]` reads `params` off `Point`'s own field labels instead of
declaring `"x", "y", "z"` separately and naming them again through a lookup. The result is a `TypedFormula[Point]`, not a
bare `Formula` — the same typing `Reify.statefulVarTyped` gives a stateful cell, below:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Expr, Reify, TypedFormula, instances}
import instances.given

val richFormula: TypedFormula[Point] = Reify[Point](uniforms = Nil) { (_, point) =>
  richProgram(point)
}
```

```scala mdoc
richFormula.formula.params
```

`Point`'s field order is the binding order `params` declares on `Formula` — `Reify[Point]` reads both from the same place, so
they can't drift apart. `Reify.paramsAs[Point](p)` is still there for a formula whose params aren't all one case class, or
where `build` wants the raw lookup for other reasons.

`ClKernel.compileT` + `TypedKernel.renderUnsafeT` launch it with a `Point[Float]` directly — no `Map`, no other case
class's `Point`-shaped lookalike will typecheck here, and, for a kernel compiled with `size = 1`, no `out` array either:
the one float the launch produces comes back as the return value:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderUnsafeT}

val richResult: Float =
  ClKernel.compileT[IO, Point](richFormula, size = 1, maxBatchSize = 1).use { kernel =>
    IO { kernel.renderUnsafeT(Point[Float](3.0f, 1.0f, 4.0f)) }
  }.unsafeRunSync()
```

## Arrays of a case class

`richProgram` also runs elementwise over device-resident arrays — one array per field instead of one float. `Reify.arraysTyped[Point]`
is the array analogue of `Reify[Point]`: it declares one input per field of `Point` and hands `build` the filled `Point[Expr]`, so the
same `richProgram(point)` reifies unchanged:

```scala mdoc:silent
val richArrayFormula: TypedFormula[Point] = Reify.arraysTyped[Point](uniforms = Nil, params = Nil) { (_, _, point) =>
  richProgram(point)
}
```

```scala mdoc
richArrayFormula.formula.inputs
```

`TypedKernel.writeInputsT` takes a `Point[Array[Float]]` — one array per field, matched by name — instead of one
`writeInputUnsafe` call per array:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, writeInputsT}

val n = 1024
val richArrayResult: Array[Float] =
  ClKernel.compileT[IO, Point](richArrayFormula, size = n, maxBatchSize = 1).use { kernel =>
    IO {
      kernel.writeInputsT(Point[Array[Float]](
        x = Array.fill(n)(3.0f),
        y = Array.fill(n)(1.0f),
        z = Array.fill(n)(4.0f),
      ))
      val out = new Array[Float](n)
      kernel.kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
      out
    }
  }.unsafeRunSync()
```

Every element answers the same as the single-point launch above, since every element was given the same `x`, `y`, `z`.

## Stateful cells

`Var[F, V, A]` describes one cell update; `.at(id)` places it in the formula state store. Here the cell is an
exponential moving average: outputs the previous value, stores the blended next one.

`alpha` is a uniform, `x` a parameter, declared together as one case class rather than two `List[String]`s plus lookups.
`Reify.statefulVarTyped[Args]` needs `uniformFields = Set("alpha")` to sort the rest, and hands `program` the whole
`Args[Expr]` back filled, instead of one lookup per role:

```scala mdoc:silent
import cats.Id
import cats.data.StateT
import io.github.mercurievv.spireopencl.symbolic.{Expr, TypedFormula}
import io.github.mercurievv.spireopencl.symbolic.state.{Store, Var, at}

case class Args[T](alpha: T, x: T)

def smooth(args: Args[Expr]): Var[Id, Expr, Expr] =
  StateT { prev =>
    val next = prev + (args.x - prev) * args.alpha
    (next, prev)
  }

val statefulFormula: TypedFormula[Args] = Reify.statefulVarTyped[Args](uniformFields = Set("alpha")) { args =>
  smooth(args).at[Store[Expr]](0) // cell id 0; use distinct ids for independent cells.
}
```

```scala mdoc
statefulFormula.formula.states
```

`ClKernel.compileT` keeps `Args` attached to the compiled kernel, so `TypedKernel.renderUnsafeT` only accepts an
`Args[Float]` — a case class from another formula, even sharing a field name, is a compile error here, not a launch
that silently reads the wrong value. This kernel is also `size = 1`, so each launch's answer comes back directly,
with no `out` array to allocate or read:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderUnsafeT}

val ema: Vector[Float] =
  ClKernel.compileT[IO, Args](statefulFormula, size = 1).use { kernel =>
    IO { Vector.fill(5)(1.0f).map(sample => kernel.renderUnsafeT(Args[Float](alpha = 0.5f, x = sample))) }
  }.unsafeRunSync()
```

Expected sequence:

```scala mdoc
Vector.iterate(0.0f, 5)(prev => prev + (1.0f - prev) * 0.5f)
```

The first output is the initial state. Each launch writes the next state back to the device buffer.
