package io.github.mercurievv.spireopencl.bench

import _root_.algebra.ring.Field
import spire.algebra.{NRoot, Trig}
import spire.implicits.*
import spire.math.{exp, sin, sqrt}

/** The benchmarked arithmetic, written once.
  *
  * Every contender runs *this* source. Instantiated at `Expr` it is the kernel; instantiated at `Float` it is the Spire-on-JVM row; the Breeze and
  * while-loop rows hand-transcribe the same expression, which is what `AgreementSpec` exists to police. Writing the GPU and CPU sides separately is
  * the standard way a benchmark ends up comparing two different computations, and the whole claim of this library is that it does not have to.
  *
  * `depth` is a parameter rather than a constant because the arithmetic-per-byte ratio is the axis that separates transfer cost from compute cost: at
  * a fixed array size, time against depth is a line whose intercept is transfer plus launch latency and whose slope is arithmetic.
  */
object Programs:

  /** Memory-bound: a fused multiply-add chain, one dependent multiply and add per level, nothing transcendental. At depth 1 this is `a * x + b` —
    * saxpy's inner expression.
    */
  def chain[V: Field](x: V, a: V, b: V, depth: Int): V =
    var acc = x
    var k = 0
    while k < depth do
      acc = acc * a + b
      k += 1
    acc

  /** Compute-bound: transcendentals dominate and the array traffic is unchanged, so this is the same memory cost as `chain` at depth 1 with far more
    * work per element. On a GPU the transcendentals are hardware; on the JVM they are `StrictMath` calls.
    */
  def heavy[V: {Field, Trig, NRoot}](x: V, a: V): V =
    val s = sin(x * a)
    val e = exp(s)
    sqrt(e * e + s * s)

  /** The elementwise workload of encoding B: three inputs per element, one output. */
  def elementwise[V: Field](a: V, b: V, c: V): V = a * b + c
