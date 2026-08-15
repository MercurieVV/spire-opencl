package io.github.mercurievv.spireopencl.bench

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, Kernel}
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit

/** The compute-versus-transfer separation, obtained by varying arithmetic at a fixed array size.
  *
  * `onPhase` cannot give it: the upload is enqueued `CL_FALSE` and the NDRange asynchronously, so the phase timestamps measure driver calls returning
  * and nearly all device time is absorbed by the blocking readback. What can be measured instead is the *slope*. Every depth here moves exactly the
  * same bytes — one output array, no inputs — and differs only in multiply-adds per element. Plot time against depth: the intercept is transfer plus
  * launch latency, the slope is arithmetic. Run the same sweep on the JVM rows and the two slopes are the throughput ratio, with all fixed overhead
  * divided out.
  *
  * Fixed at 10^6, between the size where launch latency is everything and the size where the 40 MB readback is.
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
class DepthSweepBench:

  private var counters: PhaseCounters = null

  private val hook: (String, Long) => Unit = (phase, at) => counters.record(phase, at)

  @Benchmark
  def opencl(k: DepthKernel, in: DepthInputs, ph: PhaseCounters, bh: Blackhole): Unit =
    counters = ph
    ph.begin(System.nanoTime())
    k.kernel.renderBatchIntoUnsafe(in.uniforms, GeneratorBench.OneElement, k.native, hook)
    bh.consume(k.native)

  @Benchmark
  def spire(in: DepthInputs, bh: Blackhole): Unit =
    Jvm.spireGeneratorChain(Bench.UniformA, Bench.UniformB, in.depth, in.out)
    bh.consume(in.out)

  @Benchmark
  def plain(in: DepthInputs, bh: Blackhole): Unit =
    Jvm.plainGeneratorChain(Bench.UniformA, Bench.UniformB, in.depth, in.out)
    bh.consume(in.out)

/** The array size the sweep holds fixed. The depths themselves have to be repeated as `@Param` literals below — an annotation argument must be a
  * constant expression, so they cannot be named here.
  */
object DepthSweep:
  final val Size = 1000000

@State(Scope.Thread)
class DepthInputs:

  @Param(Array("1", "8", "32", "128"))
  var depth: Int = 0

  var out: Array[Float] = null
  var uniforms: Map[String, Float] = Map.empty

  @Setup(Level.Trial)
  def setup(): Unit =
    out      = new Array[Float](DepthSweep.Size)
    uniforms = Map("a" -> Bench.UniformA, "b" -> Bench.UniformB)

/** One kernel per depth: the depth is baked into the expression tree at reification, so it is a compile-time property of the kernel and not an
  * argument. That is the point of the sweep — `nodeCount` grows with depth and the generated source grows with it.
  */
@State(Scope.Benchmark)
class DepthKernel:

  @Param(Array("1", "8", "32", "128"))
  var depth: Int = 0

  var kernel: Kernel[IO] = null
  var native: FloatBuffer = null

  private var release: IO[Unit] = IO.unit

  @Setup(Level.Trial)
  def setup(): Unit =
    val (k, r) = ClKernel.compile[IO](Formulas.generatorChain(depth), DepthSweep.Size, 1).allocated.unsafeRunSync()
    kernel  = k
    release = r
    native  = Data.directFloats(DepthSweep.Size)

  @TearDown(Level.Trial)
  def teardown(): Unit =
    release.unsafeRunSync()
    kernel = null
    native = null
