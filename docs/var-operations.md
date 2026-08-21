# State and More Operations

Two features beyond the first kernel: a wider Spire surface and a small stateful cell. Every launch here is typed on both
sides — an `In[_]` case class of arguments goes in, an `Out[_]` case class of results comes out, both fixed at compile time
to the formula that declared them. mdoc-checked.

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

`richProgram` returns one `V`, so its `Out` is a one-field case class — `value` is as good a name as any, since a single
result carries none of the naming information several results would:

```scala mdoc
case class Answer[V](value: V)
```

Reifying it still needs one `Expr` per field, but `Reify.outTyped[Point, Answer]` reads both `In`'s params and `Out`'s result
names off the case classes' own field labels instead of declaring `"x", "y", "z"` as a list and naming the result again
through a lookup:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Expr, Reify, TypedFormula, TypedFormula2, instances}
import instances.given

val richFormula: TypedFormula2[Point, Answer] = Reify.outTyped[Point, Answer](uniforms = Nil) { (_, point) =>
  Answer(richProgram(point))
}
```

```scala mdoc
richFormula.formula.params
richFormula.outputNames
```

`Point`'s field order is the binding order `params` declares on `Formula`, and `Answer`'s is `outputNames` — `Reify.outTyped`
reads all three from the case classes themselves, so they can't drift apart. `Reify.paramsAs[Point](p)` is still there for a
formula whose params aren't all one case class, or where `build` wants the raw lookup for other reasons.

`ClKernel.compileT2` + `TypedKernel2.renderT` launch it with a `Point[Float]` and hand back an `Answer[Float]` — no `Map`
on the way in, no `out` array to allocate and index on the way out, and no other case class on either side, `Point`- or
`Answer`-shaped, will typecheck here:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderT}

val richResult: Answer[Float] =
  ClKernel.compileT2[IO, Point, Answer](richFormula, size = 1, maxBatchSize = 1).use { kernel =>
    IO { kernel.renderT(Point[Float](3.0f, 1.0f, 4.0f)) }
  }.unsafeRunSync()
```

## Arrays of a case class

`richProgram` also runs elementwise over device-resident arrays — one array per field instead of one float. `Out` stays implicit
here: the launch answers with the whole array it computed, one value per array element, rather than a single named result, so
there is nothing for an `Out` case class to name. `In` stays exactly as typed as before — `Reify.arraysTyped[Point]` is the array
analogue of `Reify[Point]`: it declares one input per field of `Point` and hands `build` the filled `Point[Expr]`, so the same
`richProgram(point)` reifies unchanged:

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

## Multiple outputs

`Answer` above has one field because `richProgram` returns one `V`. Nothing about `Reify.outTyped` requires that — `Out` is
just a case class, and a formula that computes more than one result in the same pass over its arguments declares them all:

```scala mdoc
case class Result[V](sum: V, product: V)

def stats[V: Field](point: Point[V]): Result[V] =
  Result(sum = point.x + point.y + point.z, product = point.x * point.y * point.z)

stats(Point[Double](3.0, 1.0, 4.0))
```

Reifying it is the same call as `richFormula` above, `Out` swapped for `Result`:

```scala mdoc:silent
val statsFormula: TypedFormula2[Point, Result] = Reify.outTyped[Point, Result](uniforms = Nil) { (_, point) =>
  stats(point)
}
```

```scala mdoc
statsFormula.outputNames
```

`ClKernel.compileT2` + `TypedKernel2.renderT` launch it exactly like `richFormula` above — `Point[Float]` in, `Out[Float]` out —
except this `Out[Float]` is a `Result[Float]` with two fields instead of `Answer[Float]`'s one, both produced by the same
launch:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderT}

val statsResult: Result[Float] =
  ClKernel.compileT2[IO, Point, Result](statsFormula, size = 1, maxBatchSize = 1).use { kernel =>
    IO { kernel.renderT(Point[Float](3.0f, 1.0f, 4.0f)) }
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

`Reify.statefulVarTyped` predates `Reify.outTyped` and only types `In`, not `Out` — there is no `Var`/`.at` program that returns
more than one cell's worth of output, so `TypedFormula[Args]` never needed an `Out` of its own. `TypedKernel.renderT[Answer]`
closes that gap from the launch side instead: it wraps `TypedKernel.renderUnsafeT`'s bare `Float` in the same one-field
`Answer[Float]` `richFormula` above returns, so every typed launch in this library answers `Out[Float]`, not a mix of `Float`
and `Out[Float]` depending on which formula built the kernel. `ClKernel.compileT` still keeps `Args` attached to the compiled
kernel, so a case class from another formula, even sharing a field name, is a compile error here, not a launch that silently
reads the wrong value:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, renderT}

val ema: Vector[Answer[Float]] =
  ClKernel.compileT[IO, Args](statefulFormula, size = 1).use { kernel =>
    IO { Vector.fill(5)(1.0f).map(sample => kernel.renderT[Answer](Args[Float](alpha = 0.5f, x = sample))) }
  }.unsafeRunSync()
```

Expected sequence:

```scala mdoc
Vector.iterate(0.0f, 5)(prev => prev + (1.0f - prev) * 0.5f)
```

The first output is the initial state. Each launch writes the next state back to the device buffer.
