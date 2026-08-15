package io.github.mercurievv.spireopencl.bench

import breeze.linalg.DenseVector
import spire.implicits.*

/** The three JVM contenders.
  *
  * `spire*` calls `Programs` at `Float` — the same source the kernel was reified from. `plain*` hand-writes the identical expression as a `while`
  * loop over primitive arrays: it is the floor, and without it a ratio between the other rows has no scale. `breeze*` uses `DenseVector[Float]`.
  *
  * Every method writes into a caller-owned array. Allocating the result inside would make the 10^7 rows a measurement of the allocator.
  */
object Jvm:

  // ---- Encoding A: value is a function of the index, no input arrays ----

  def spireGeneratorChain(a: Float, b: Float, depth: Int, out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      out(i) = Programs.chain[Float](i.toFloat, a, b, depth)
      i += 1

  def plainGeneratorChain(a: Float, b: Float, depth: Int, out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      var acc = i.toFloat
      var k = 0
      while k < depth do
        acc = acc * a + b
        k += 1
      out(i) = acc
      i += 1

  def spireGeneratorHeavy(a: Float, out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      out(i) = Programs.heavy[Float](i.toFloat, a)
      i += 1

  def plainGeneratorHeavy(a: Float, out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      val s = math.sin(i.toFloat * a).toFloat
      val e = math.exp(s).toFloat
      out(i) = math.sqrt((e * e + s * s).toDouble).toFloat
      i += 1

  /** Breeze's generator row. `DenseVector.tabulate` is the idiomatic way to build a vector from its index, and it is what a Breeze user would write;
    * there is no fused-multiply-add vector primitive to reach for, so the chain stays scalar inside the tabulate.
    *
    * It allocates a fresh vector per call, unlike every other row. That is not harness sloppiness — it is Breeze's API: `tabulate` has no in-place
    * form. The allocation is part of what using Breeze this way costs and is left in deliberately, noted here so the number is read correctly.
    */
  def breezeGeneratorChain(a: Float, b: Float, depth: Int, n: Int): DenseVector[Float] =
    DenseVector.tabulate(n)(i => Programs.chain[Float](i.toFloat, a, b, depth))

  // ---- Encoding B: elementwise over three input arrays ----

  def spireElementwise(a: Array[Float], b: Array[Float], c: Array[Float], out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      out(i) = Programs.elementwise[Float](a(i), b(i), c(i))
      i += 1

  def plainElementwise(a: Array[Float], b: Array[Float], c: Array[Float], out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      out(i) = a(i) * b(i) + c(i)
      i += 1

  def spireElementwiseHeavy(a: Array[Float], b: Array[Float], c: Array[Float], out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      out(i) = Programs.heavy[Float](a(i), b(i)) * c(i)
      i += 1

  def plainElementwiseHeavy(a: Array[Float], b: Array[Float], c: Array[Float], out: Array[Float]): Unit =
    var i = 0
    while i < out.length do
      val s = math.sin(a(i) * b(i)).toFloat
      val e = math.exp(s).toFloat
      out(i) = math.sqrt((e * e + s * s).toDouble).toFloat * c(i)
      i += 1

  /** Breeze at its best on this workload: `a *:* b + c` is fully vectorised through Breeze's `DenseVector[Float]` ufuncs, no per-element closure.
    *
    * Allocates two intermediates (the product, then the sum), which is exactly what the idiomatic expression costs. An in-place formulation exists
    * but is not what anyone writes.
    */
  def breezeElementwise(a: DenseVector[Float], b: DenseVector[Float], c: DenseVector[Float]): DenseVector[Float] =
    (a *:* b) + c

  /** Breeze's gap. `breeze.numerics.sin` / `exp` / `sqrt` are implemented for `DenseVector[Double]`, not `DenseVector[Float]`, so the compute-bound
    * workload has no vectorised Breeze form at this precision. Rather than silently promote to `Double` — which would compare a different computation
    * — the row falls back to a per-element map, and the benchmark reports it as the scalar fallback it is.
    */
  def breezeElementwiseHeavy(a: DenseVector[Float], b: DenseVector[Float], c: DenseVector[Float]): DenseVector[Float] =
    DenseVector.tabulate(a.length)(i => Programs.heavy[Float](a(i), b(i)) * c(i))
