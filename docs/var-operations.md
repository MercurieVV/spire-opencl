# State and More Operations

Every launch is typed on both sides: an `In[_]` case class of arguments goes in, an `Out[_]` case class of results comes
out, both fixed at compile time to the formula. mdoc-checked.

## More Spire operations

```scala mdoc
import spire.algebra.{Field, NRoot, Trig}
import spire.implicits.*
import spire.math.{sin, sqrt}

case class Point[V](x: V, y: V, z: V)
case class Answer[V](value: V)

def richProgram[V: {Field, NRoot, Trig}](point: Point[V]): Answer[V] =
  val ratio = (point.x - point.y) / (point.z + 1)
  Answer(sqrt(ratio * ratio) + sin(point.y))

richProgram(Point[Double](3.0, 1.0, 4.0))
```

`Reify.outTyped[In, Out]` reads `params` off `In`'s field labels and result names off `Out`'s, instead of declaring them
as lists:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Expr, Reify, TypedFormula, TypedFormula2, instances}
import instances.given

val richFormula: TypedFormula2[Point, Answer] = Reify.outTyped[Point, Answer](uniforms = Nil) { (_, point) =>
  richProgram(point)
}
```

```scala mdoc
richFormula.formula.params
richFormula.outputNames
```

`ClKernel.compileT2` + `TypedKernel2.renderT` take a `Point[Float]` and return an `Answer[Float]` — no `Map`, no `out`
array:

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

Elementwise over device-resident arrays instead of one value: `Out` has no case class here — the answer is the whole
array. `In` is still typed via `Reify.arraysTyped[Point]`, one input per field:

```scala mdoc:silent
val richArrayFormula: TypedFormula[Point] = Reify.arraysTyped[Point](uniforms = Nil, params = Nil) { (_, _, point) =>
  richProgram(point).value
}
```

```scala mdoc
richArrayFormula.formula.inputs
```

`TypedKernel.writeInputsT` takes a `Point[Array[Float]]` — one array per field, matched by name:

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

## Multiple outputs

`Out` is just a case class — nothing limits it to one field:

```scala mdoc
case class Result[V](sum: V, product: V)

def stats[V: Field](point: Point[V]): Result[V] =
  Result(sum = point.x + point.y + point.z, product = point.x * point.y * point.z)

stats(Point[Double](3.0, 1.0, 4.0))
```

```scala mdoc:silent
val statsFormula: TypedFormula2[Point, Result] = Reify.outTyped[Point, Result](uniforms = Nil) { (_, point) =>
  stats(point)
}
```

```scala mdoc
statsFormula.outputNames
```

Same launch shape as `richFormula` above, both results from one launch:

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

`Var[F, V, A]` describes one cell update; `.at(id)` places it in the formula state store. Here: an exponential moving
average, outputting the previous value and storing the blended next one.

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

`Reify.statefulVarTyped` only types `In`; `TypedKernel.renderT[Answer]` wraps its bare `Float` output in the same
`Answer` used above, so every typed launch answers `Out[Float]`:

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

The first output is the initial state; each launch writes the next state back to the device buffer.
