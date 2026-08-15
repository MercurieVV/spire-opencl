# State and More Operations

This page shows two features beyond the first kernel: a wider Spire surface and a small stateful cell.
The examples are mdoc-checked.

## More Spire operations

`Expr` has instances for common Spire typeclasses:

```scala mdoc
import spire.algebra.{Field, NRoot, Trig}
import spire.implicits.*
import spire.math.{sin, sqrt}

def richProgram[V: {Field, NRoot, Trig}](x: V, y: V, z: V): V =
  val ratio = (x - y) / (z + 1)
  sqrt(ratio * ratio) + sin(y)

richProgram[Double](3.0, 1.0, 4.0)
```

The same function can be reified:

```scala mdoc:silent
import io.github.mercurievv.spireopencl.symbolic.{Reify, instances}
import instances.given

val richFormula = Reify(uniforms = Nil, params = List("x", "y", "z")) { (_, p) =>
  richProgram(p("x"), p("y"), p("z"))
}
```

```scala mdoc
richFormula.params
```

## Stateful cells

`Var[F, V, A]` describes one cell update. `.at(id)` places that cell in the formula state store.
Here the cell is an exponential moving average: it outputs the previous value, then stores the blended
next value.

```scala mdoc:silent
import cats.Id
import cats.data.StateT
import io.github.mercurievv.spireopencl.symbolic.Expr
import io.github.mercurievv.spireopencl.symbolic.state.{Store, Var, at}

val alpha = Expr.Uniform("alpha")

val smooth: Var[Id, Expr, Expr] =
  StateT { prev =>
    val sample = Expr.Param("x")
    val next = prev + (sample - prev) * alpha
    (next, prev)
  }

val statefulFormula = Reify.statefulVar(uniforms = List("alpha"), params = List("x")) { (_, _) =>
  smooth.at[Store[Expr]](0) // cell id 0; use distinct ids for independent cells.
}
```

```scala mdoc
statefulFormula.states
```

Hardware launch, compile-checked by mdoc:

```scala mdoc:compile-only
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.ClKernel

val ema: Vector[Float] =
  ClKernel.compile[IO](statefulFormula, size = 1).use { kernel =>
    IO {
      val out = new Array[Float](1)
      Vector.fill(5)(1.0f).map { sample =>
        kernel.renderUnsafe(Map("alpha" -> 0.5f), Map("x" -> sample), out)
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
