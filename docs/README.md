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

## Arrays

The values above are scalars: one per launch, or one per batch element. An **input** is one value per
work-item — an array, living in device memory, indexed by the work-item's position. That is what
ordinary array code needs, and `Reify.arrays` declares them:

```scala mdoc
def elementwise[V: Field](a: V, b: V, c: V): V = a * b + c

val arrayFormula = Reify.arrays(uniforms = Nil, params = Nil, inputs = List("a", "b", "c")) {
  (_, _, in) => elementwise(in("a"), in("b"), in("c"))
}
```

Each input becomes its own `__global const float*` argument, read at `a[i]`:

```scala mdoc
println(io.github.mercurievv.spireopencl.opencl.CodeGen(arrayFormula).linesIterator.next())
```

Writing an array and launching are **separate calls**, which is the point. The array stays on the
device and is read by every launch until it is written again, so a caller whose data does not change
between launches transfers it once rather than once per launch — normally the dominant cost:

```scala mdoc:compile-only
val n = 1000000
ClKernel.compile[IO](arrayFormula, size = n, maxBatchSize = 1).use { kernel =>
  IO {
    kernel.writeInputUnsafe("a", Array.fill(n)(1.5f))    // uploaded once...
    kernel.writeInputUnsafe("b", Array.fill(n)(2.0f))
    kernel.writeInputUnsafe("c", Array.fill(n)(0.25f))
    val out = new Array[Float](n)
    kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)   // ...read by this launch
    kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)   // ...and by this one, transferring nothing
    out(0)                                               // 3.25f
  }
}
```

Launching before an array has been written is refused rather than answered with zeros. Inputs compose
with the other kinds: an input varies along dimension 0 and a parameter along dimension 1, so one
launch can apply several per-element parameters to the same resident array.

Once the inputs stay resident, the only transfer a launch still has is the results coming back. That
one can go too — `renderBatchMappedUnsafe` asks the driver for the results where they already are
rather than copying them into caller memory:

```scala mdoc:compile-only
val n = 1000000
ClKernel.compile[IO](arrayFormula, size = n, maxBatchSize = 1, hostVisibleOutput = true).use { kernel =>
  IO {
    kernel.writeInputUnsafe("a", Array.fill(n)(1.5f))
    kernel.writeInputUnsafe("b", Array.fill(n)(2.0f))
    kernel.writeInputUnsafe("c", Array.fill(n)(0.25f))
    kernel.renderBatchMappedUnsafe(Map.empty, Seq(Map.empty)) { results =>
      results.get(0)                       // valid only inside this block
    }
  }
}
```

The buffer is unmapped on the way out, so read what is needed — or copy it — before returning.
`hostVisibleOutput` is opt-in because the allocation flag that makes mapping cheap is not free on
every device.
