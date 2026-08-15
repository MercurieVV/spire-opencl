package io.github.mercurievv.spireopencl.bench

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.util.concurrent.TimeUnit

/** Encoding **B**: `a * b + c` over three host arrays — the workload everyone means by "elementwise on big arrays", and the one this library has no
  * native shape for.
  *
  * There is no array kernel argument, so the array has to run along the *batch* dimension with a single work-item in dimension 0, and the three
  * inputs arrive element-major in one packed parameter buffer. Both the upload and the readback are on the measured path, which is the honest
  * accounting: a caller with arrays in JVM memory pays them every call.
  *
  * `openclPacked` starts from data already interleaved; `openclPacking` includes the interleave. A caller holding three parallel arrays pays the
  * second number, not the first.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Threads(1)
@Fork(
  value         = 3,
  jvmArgsAppend = Array("-Xms8g", "-Xmx8g", "-XX:+AlwaysPreTouch", "-XX:MaxDirectMemorySize=2g"),
)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Thread)
class ElementwiseBench:

  private var counters: PhaseCounters = null

  private val hook: (String, Long) => Unit = (phase, at) => counters.record(phase, at)

  // ---- spire-opencl ----

  @Benchmark
  def openclPacked(k: ElementwiseKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.plain.renderBatchPackedUnsafe(Map.empty, in.packed, in.size, k.native, hook)
    bh.consume(k.native)

  @Benchmark
  def openclPacking(k: ElementwiseKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    Data.interleave(in.a, in.b, in.c, in.packed)
    k.plain.renderBatchPackedUnsafe(Map.empty, in.packed, in.size, k.native, hook)
    bh.consume(k.native)

  @Benchmark
  def openclHeavy(k: ElementwiseKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.heavy.renderBatchPackedUnsafe(Map.empty, in.packed, in.size, k.native, hook)
    bh.consume(k.native)

  // ---- spire-opencl, encoding C: arrays resident on the device ----

  @Benchmark
  def openclInput(k: InputKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.plain.renderBatchIntoUnsafe(Map.empty, GeneratorBench.OneElement, k.native, hook)
    bh.consume(k.native)

  @Benchmark
  def openclInputHeavy(k: InputKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.heavy.renderBatchIntoUnsafe(Map.empty, GeneratorBench.OneElement, k.native, hook)
    bh.consume(k.native)

  /** Encoding C with the copy removed too: arrays resident, results left where the device wrote them.
    *
    * Paired with `openclInput` above, and deliberately symmetric with it — both rows launch and leave the caller holding a buffer of results, and
    * neither reads through it. The difference between the two is the device-to-host copy and nothing else, which is the question mapping answers.
    *
    * What a caller then does with the buffer is the caller's cost either way, and it is a large one: a `FloatBuffer.get` loop over 10^7 floats runs
    * about 8 ms, four times the launch. Folding that into these rows would bury the thing being measured.
    */
  @Benchmark
  def openclInputMapped(k: InputKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    bh.consume(k.plain.renderBatchMappedUnsafe(Map.empty, GeneratorBench.OneElement, hook)(identity))

  // ---- Spire typeclasses over primitive arrays ----

  @Benchmark
  def spireElementwise(in: Inputs, bh: Blackhole): Unit =
    Jvm.spireElementwise(in.a, in.b, in.c, in.out)
    bh.consume(in.out)

  @Benchmark
  def spireHeavy(in: Inputs, bh: Blackhole): Unit =
    Jvm.spireElementwiseHeavy(in.a, in.b, in.c, in.out)
    bh.consume(in.out)

  // ---- Breeze ----

  @Benchmark
  def breezeElementwise(in: Inputs, bh: Blackhole): Unit =
    bh.consume(Jvm.breezeElementwise(in.ba, in.bb, in.bc))

  /** Breeze's gap, benchmarked as the scalar fallback it is: `breeze.numerics.sin`/`exp`/`sqrt` have no `DenseVector[Float]` implementation, so there
    * is no vectorised Breeze form of the compute-bound program at this precision. Promoting to `Double` would compare a different computation.
    */
  @Benchmark
  def breezeHeavy(in: Inputs, bh: Blackhole): Unit =
    bh.consume(Jvm.breezeElementwiseHeavy(in.ba, in.bb, in.bc))

  // ---- The floor ----

  @Benchmark
  def plainElementwise(in: Inputs, bh: Blackhole): Unit =
    Jvm.plainElementwise(in.a, in.b, in.c, in.out)
    bh.consume(in.out)

  @Benchmark
  def plainHeavy(in: Inputs, bh: Blackhole): Unit =
    Jvm.plainElementwiseHeavy(in.a, in.b, in.c, in.out)
    bh.consume(in.out)
