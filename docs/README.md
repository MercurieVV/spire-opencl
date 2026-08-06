# spire-opencl

Instantiate numeric code at a symbolic value type, get an OpenCL kernel.

The code fences below marked `mdoc` are compiled (and, where safe without a real device, executed)
by `mill docs.mdoc` on every run — this file cannot drift from what the library actually accepts.

## The idea

Write the program in spire. Say what its arguments are. Run it on the GPU.

```scala mdoc
import spire.algebra.{Field, Trig}
import spire.implicits.*
import spire.math.sin

// Ordinary spire. Nothing here imports this library or knows a GPU exists.
def program[V: {Field, Trig}](b: V, c: V, d: V): V = b * c - sin(d)

// V = Double here: an ordinary numeric result, computed with no GPU involved.
program[Double](2.5, 4.0, 0.75)
```

```scala mdoc
import cats.effect.IO
import io.github.mercurievv.spireopencl.symbolic.{Reify, instances}
import io.github.mercurievv.spireopencl.opencl.ClKernel
import instances.given

val formula = Reify(uniforms = Nil, params = List("b", "c", "d")) { (_, arg) =>
  program(arg("b"), arg("c"), arg("d"))     // V = Expr, so this builds the tree
}
```

`program` is called exactly the same way in both instantiations. At `V = Double` it computes a number;
at `V = Expr` the arithmetic *is* the tree, because `Field[Expr]` and `Trig[Expr]` build nodes instead
of folding numbers. That tree is the IR, and an IR can be compiled.

Running the compiled kernel needs an actual OpenCL device, so that step is `compile-only` here — mdoc
still type-checks it against the real API on every run, it just doesn't execute against hardware:

```scala mdoc:compile-only
ClKernel.compile[IO](formula, size = 1, maxBatchSize = 1).use { kernel =>
  IO {
    val out = new Array[Float](1)
    kernel.renderUnsafe(Map.empty, Map("b" -> 2.5f, "c" -> 4.0f, "d" -> 0.75f), out)
    println(out(0))                         // 9.318..., same as program[Double](2.5, 4.0, 0.75)
    out(0)
  }
}
```

Reification is by **application**, not inspection: the function is applied to symbolic inputs and the
result is kept. No macros, no free-arrow encoding, no DSL to learn.
