package io.github.mercurievv.spireopencl.bench

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.util.concurrent.TimeUnit

/** Encoding **A**: the value of every element is a function of its index, which is the shape `spire-opencl` was built for.
  *
  * Nothing is uploaded — two scalar uniforms and a launch — so the GPU's only transfer is the result coming back. This is the library's best case and
  * the fairest reading of what it offers. The JVM rows fill the same `size` floats from the same expression.
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
class GeneratorBench:

  /** The `onPhase` hook, allocated once. It reads the counters state through a field rewritten at the top of each launch rather than capturing one,
    * because JMH owns the state's lifecycle and hands it to the method, not to the constructor.
    */
  private var counters: PhaseCounters = null

  private val hook: (String, Long) => Unit = (phase, at) => counters.record(phase, at)

  // ---- spire-opencl ----

  @Benchmark
  def openclChain(k: GeneratorKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.plain.renderBatchIntoUnsafe(in.uniforms, GeneratorBench.OneElement, k.native, hook)
    bh.consume(k.native)

  @Benchmark
  def openclHeavy(k: GeneratorKernels, in: Inputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.heavy.renderBatchIntoUnsafe(in.uniforms, GeneratorBench.OneElement, k.native, hook)
    bh.consume(k.native)

  // ---- Spire typeclasses over a primitive array ----

  @Benchmark
  def spireChain(in: Inputs, bh: Blackhole): Unit =
    Jvm.spireGeneratorChain(in.uniformA, in.uniformB, Bench.Depth, in.out)
    bh.consume(in.out)

  @Benchmark
  def spireHeavy(in: Inputs, bh: Blackhole): Unit =
    Jvm.spireGeneratorHeavy(in.uniformA, in.out)
    bh.consume(in.out)

  // ---- Breeze ----

  @Benchmark
  def breezeChain(in: Inputs, bh: Blackhole): Unit =
    bh.consume(Jvm.breezeGeneratorChain(in.uniformA, in.uniformB, Bench.Depth, in.size))

  // ---- The floor: a while loop over a primitive array ----

  @Benchmark
  def plainChain(in: Inputs, bh: Blackhole): Unit =
    Jvm.plainGeneratorChain(in.uniformA, in.uniformB, Bench.Depth, in.out)
    bh.consume(in.out)

  @Benchmark
  def plainHeavy(in: Inputs, bh: Blackhole): Unit =
    Jvm.plainGeneratorHeavy(in.uniformA, in.out)
    bh.consume(in.out)

object GeneratorBench:
  /** One batch element with no parameters: encoding A puts the whole array in dimension 0, so a launch carries a single element and its `Map` is
    * empty. Hoisted to a constant because building it per invocation would allocate inside the measured path.
    */
  val OneElement: Seq[Map[String, Float]] = Seq(Map.empty)
