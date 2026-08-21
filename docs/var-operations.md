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
declaring `"x", "y", "z"` separately and naming them again through a lookup:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify, instances}
import instances.given

val richFormula: Formula = Reify[Point](uniforms = Nil) { (_, point) =>
  richProgram(point)
}
```

```scala mdoc
richFormula.params
```

`Point`'s field order is the binding order `params` declares on `Formula` — `Reify[Point]` reads both from the same place, so
they can't drift apart. `Reify.paramsAs[Point](p)` is still there for a formula whose params aren't all one case class, or
where `build` wants the raw lookup for other reasons.

`Reify[Point]` still returns a bare `Formula`, so it launches like any other — params by name, no `Args` type pin:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.ClKernel

val richResult: Float =
  ClKernel.compile[IO](richFormula, size = 1, maxBatchSize = 1).use { kernel =>
    IO {
      val out = new Array[Float](1)
      kernel.renderUnsafe(
        uniforms = Map.empty,
        params = Map("x" -> 3.0f, "y" -> 1.0f, "z" -> 4.0f),
        out = out,
      )
      out(0)
    }
  }.unsafeRunSync()
```

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
that silently reads the wrong value:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderUnsafeT}

val ema: Vector[Float] =
  ClKernel.compileT[IO, Args](statefulFormula, size = 1).use { kernel =>
    IO {
      val out = new Array[Float](1)
      Vector.fill(5)(1.0f).map { sample =>
        kernel.renderUnsafeT(Args[Float](alpha = 0.5f, x = sample), out)
        out(0)
      }
    }
  }.unsafeRunSync()
```

Expected sequence:

```scala mdoc
Vector.iterate(0.0f, 5)(prev => prev + (1.0f - prev) * 0.5f)
```

The first output is the initial state. Each launch writes the next state back to the device buffer.
