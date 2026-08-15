package io.github.mercurievv.spireopencl.bench

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, Kernel}
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit

/** Latency per launch against throughput across many.
  *
  * Every blocking entry point pays a full host round trip to the device, and at small sizes that round trip is not part of the cost — it *is* the
  * cost. Every `opencl` row in this suite has a floor near 180 us that no amount of arithmetic moves, because the arithmetic was never what it was
  * waiting for.
  *
  * `blocking` is that: `launches` launches, each waiting for its own results. `pipelined` enqueues all of them, drains the queue once, and collects
  * afterwards. The work is identical; only the number of times the host stops and waits differs. Divide either score by `launches` for the per-launch
  * cost.
  *
  * The batch dimension already amortises this within one launch, which is why it exists. What it cannot do is amortise across launches that are not
  * known at the same time — a caller producing blocks in sequence has to enqueue them as they arrive.
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
class PipelineBench:

  @Benchmark
  def blocking(s: Pipeline, bh: Blackhole): Unit =
    var k = 0
    while k < s.launches do
      s.kernel.renderBatchIntoUnsafe(Map.empty, GeneratorBench.OneElement, s.native)
      k += 1
    bh.consume(s.native)

  @Benchmark
  def pipelined(s: Pipeline, bh: Blackhole): Unit =
    var k = 0
    while k < s.launches do
      val slot = s.kernel.enqueueBatchUnsafe(Map.empty, GeneratorBench.OneElement)
      // The readback is enqueued too, into that slot's own destination. Collecting with blocking reads would
      // move the host round trips rather than remove them, which is the whole thing being tested.
      s.kernel.enqueueReadSlotUnsafe(slot, s.destinations(k))
      k += 1
    s.kernel.finishUnsafe()
    bh.consume(s.destinations)

object Pipeline:
  /** Small on purpose. This is a latency question, and at 10^7 the transfer dominates so completely that nothing about scheduling is visible. */
  final val Size = 10000

@State(Scope.Benchmark)
class Pipeline:

  @Param(Array("1", "4", "16", "64"))
  var launches: Int = 0

  var kernel: Kernel[IO] = null
  var native: FloatBuffer = null

  /** One per launch: an enqueued read writes whenever the driver reaches it, so two launches sharing a destination would race. */
  var destinations: Array[FloatBuffer] = null

  private var release: IO[Unit] = IO.unit

  @Setup(Level.Trial)
  def setup(): Unit =
    // One slot per launch, so nothing in flight is overwritten before it is collected.
    val (k, r) = ClKernel.compile[IO](Formulas.elementwiseInputs, Pipeline.Size, 1, outputSlots = launches).allocated.unsafeRunSync()
    kernel       = k
    release      = r
    native       = Data.directFloats(Pipeline.Size)
    destinations = Array.fill(launches)(Data.directFloats(Pipeline.Size))
    k.writeInputUnsafe("a", Data.fill(Pipeline.Size, 0x9e3779b9L))
    k.writeInputUnsafe("b", Data.fill(Pipeline.Size, 0x517cc1b7L))
    k.writeInputUnsafe("c", Data.fill(Pipeline.Size, 0x27220a95L))

  @TearDown(Level.Trial)
  def teardown(): Unit =
    release.unsafeRunSync()
    kernel = null
    native = null
