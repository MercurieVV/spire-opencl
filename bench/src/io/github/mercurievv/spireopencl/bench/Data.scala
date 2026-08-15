package io.github.mercurievv.spireopencl.bench

import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}

/** Inputs and buffers for the harness.
  *
  * Deterministic, because a benchmark that cannot be re-run on the same numbers cannot be compared against yesterday's; and non-constant, because
  * anything the JIT can prove constant it will fold out of the measured loop.
  */
object Data:

  /** xorshift64* rather than `java.util.Random`: same sequence on every JVM and every run, and cheap enough that filling 10^7 floats in `@Setup` is
    * not itself the slow part of a trial. Values land in [-1, 1) — bounded, so `exp` in `heavy` cannot overflow to infinity and turn the JVM rows
    * into a denormal benchmark.
    */
  def fill(n: Int, seed: Long): Array[Float] =
    val out = new Array[Float](n)
    var s = seed
    var i = 0
    while i < n do
      s ^= s << 13
      s ^= s >>> 7
      s ^= s << 17
      out(i) = ((s >>> 40).toFloat / 8388608.0f) - 1.0f
      i += 1
    out

  /** Element-major interleave, the layout `renderBatchPackedUnsafe` wants: element `e`'s values at `[e * 3, e * 3 + 3)`, in declared parameter order.
    *
    * Kept separate from the launch so the benchmark can charge it or not. A caller who already holds three parallel arrays must pay this; one whose
    * data is already interleaved does not. Both are real, so both are measured.
    */
  def interleave(a: Array[Float], b: Array[Float], c: Array[Float], into: Array[Float]): Unit =
    var i = 0
    val n = a.length
    while i < n do
      val o = i * 3
      into(o)     = a(i)
      into(o + 1) = b(i)
      into(o + 2) = c(i)
      i += 1

  /** `renderBatchIntoUnsafe` rejects a heap buffer — JOCL cannot take a pointer to one — so the readback destination has to be allocated direct and
    * given the platform's byte order.
    */
  def directFloats(n: Int): FloatBuffer =
    ByteBuffer.allocateDirect(n * java.lang.Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer()
