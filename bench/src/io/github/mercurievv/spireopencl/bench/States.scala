package io.github.mercurievv.spireopencl.bench

import breeze.linalg.DenseVector
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, Kernel}
import io.github.mercurievv.spireopencl.symbolic.Formula
import org.openjdk.jmh.annotations.*

import java.nio.FloatBuffer

/* JMH state.
 *
 * Every class here is top level on purpose: a class declared inside a Scala `object` compiles to an inner class carrying a reference to its enclosing
 * instance, and JMH instantiates a state through its public no-argument constructor. Nesting them would fail at run time, not at compile time.
 *
 * The kernels live in their own state classes, injected only into the benchmarks that launch them, because `@Setup(Level.Trial)` on a shared state
 * would create and destroy an OpenCL context for every Spire and Breeze trial too — a driver teardown per fork that has no business in a JVM
 * measurement. JMH distributes a `@Param` by name to every state that declares it, which is why `size` appears in each of them. */

object Bench:
  /** The arithmetic depth every `chain` row outside the depth sweep uses. Deep enough that the JVM rows are not pure memory traffic, shallow enough
    * that the GPU rows are still transfer-dominated.
    */
  final val Depth = 8

  final val UniformA = 1.0000001f
  final val UniformB = 0.5f

/** Inputs, allocated once per trial and reused by every invocation.
  *
  * At 10^7 a fresh output array per invocation would make every row a measurement of the allocator and the GC rather than of the arithmetic — and the
  * JVM rows would lose to the GPU for a reason that has nothing to do with either.
  */
@State(Scope.Thread)
class Inputs:

  @Param(Array("10000", "1000000", "10000000"))
  var size: Int = 0

  var a: Array[Float] = null
  var b: Array[Float] = null
  var c: Array[Float] = null
  var out: Array[Float] = null
  var packed: Array[Float] = null
  var ba: DenseVector[Float] = null
  var bb: DenseVector[Float] = null
  var bc: DenseVector[Float] = null
  var uniformA: Float = Bench.UniformA
  var uniformB: Float = Bench.UniformB
  var uniforms: Map[String, Float] = Map.empty

  @Setup(Level.Trial)
  def setup(): Unit =
    a      = Data.fill(size, 0x9e3779b9L)
    b      = Data.fill(size, 0x517cc1b7L)
    c      = Data.fill(size, 0x27220a95L)
    out    = new Array[Float](size)
    packed = new Array[Float](size * 3)
    Data.interleave(a, b, c, packed)
    ba       = DenseVector(a)
    bb       = DenseVector(b)
    bc       = DenseVector(c)
    uniforms = Map("a" -> uniformA, "b" -> uniformB)

/** A pair of compiled kernels plus the direct buffer they read back into, torn down with the trial.
  *
  * `Resource.allocated` rather than `use`: the kernel has to outlive the setup method, and wrapping each launch in `use` would charge a context's
  * worth of setup to every invocation. The release action is kept and run in `@TearDown`.
  */
abstract class Compiled:
  var plain: Kernel[IO] = null
  var heavy: Kernel[IO] = null
  var native: FloatBuffer = null

  private var releasePlain: IO[Unit] = IO.unit
  private var releaseHeavy: IO[Unit] = IO.unit

  protected def open(
    plainF: Formula,
    heavyF: Formula,
    size: Int,
    maxBatch: Int,
    outputFloats: Int,
    hostVisibleOutput: Boolean = false,
  ): Unit =
    val (p, rp) = ClKernel.compile[IO](plainF, size, maxBatch, hostVisibleOutput).allocated.unsafeRunSync()
    val (h, rh) = ClKernel.compile[IO](heavyF, size, maxBatch, hostVisibleOutput).allocated.unsafeRunSync()
    plain        = p
    heavy        = h
    releasePlain = rp
    releaseHeavy = rh
    native       = Data.directFloats(outputFloats)

  protected def close(): Unit =
    releaseHeavy.unsafeRunSync()
    releasePlain.unsafeRunSync()
    plain  = null
    heavy  = null
    native = null

/** Encoding **A** — `size = N` work-items in dimension 0, a single batch element, no parameter upload at all. */
@State(Scope.Benchmark)
class GeneratorKernels extends Compiled:

  @Param(Array("10000", "1000000", "10000000"))
  var size: Int = 0

  @Setup(Level.Trial)
  def setup(): Unit = open(
    Formulas.generatorChain(Bench.Depth),
    Formulas.generatorHeavy,
    size,
    maxBatch     = 1,
    outputFloats = size,
  )

  @TearDown(Level.Trial)
  def teardown(): Unit = close()

/** Encoding **C** — the arrays resident on the device, read at the work-item index, `size = N` on dimension 0.
  *
  * The three uploads happen once here in `@Setup`, deliberately *outside* the measured path, because that is the claim being tested: data that does
  * not change between launches should not be sent with every launch. A caller whose arrays do change every launch is encoding B and is measured as
  * such.
  */
@State(Scope.Benchmark)
class InputKernels extends Compiled:

  @Param(Array("10000", "1000000", "10000000"))
  var size: Int = 0

  @Setup(Level.Trial)
  def setup(): Unit =
    open(
      Formulas.elementwiseInputs,
      Formulas.elementwiseHeavyInputs,
      size,
      maxBatch     = 1,
      outputFloats = size,
      // So the same state can serve both the copying and the mapping rows; only the mapping ones use the facility.
      hostVisibleOutput = true,
    )
    val a = Data.fill(size, 0x9e3779b9L)
    val b = Data.fill(size, 0x517cc1b7L)
    val c = Data.fill(size, 0x27220a95L)
    List(plain, heavy).foreach: k =>
      k.writeInputUnsafe("a", a)
      k.writeInputUnsafe("b", b)
      k.writeInputUnsafe("c", c)

  @TearDown(Level.Trial)
  def teardown(): Unit = close()

/** Encoding **B** — `size = 1`, the array along the batch dimension, three floats per element uploaded per launch.
  *
  * `maxBatchSize = size` is what lets the whole array go in one launch, and it is also what makes this the expensive shape: `ClKernel` allocates a
  * direct staging buffer and a device parameter buffer of `3 * size` floats each at compile time, so at 10^7 the two are 120 MB apiece.
  */
@State(Scope.Benchmark)
class ElementwiseKernels extends Compiled:

  @Param(Array("10000", "1000000", "10000000"))
  var size: Int = 0

  @Setup(Level.Trial)
  def setup(): Unit = open(
    Formulas.elementwise,
    Formulas.elementwiseHeavy,
    size         = 1,
    maxBatch     = size,
    outputFloats = size,
  )

  @TearDown(Level.Trial)
  def teardown(): Unit = close()
