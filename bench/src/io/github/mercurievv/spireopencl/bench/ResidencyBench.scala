package io.github.mercurievv.spireopencl.bench

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, Kernel}
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit

/** How many launches it takes before keeping the arrays on the device pays for putting them there.
  *
  * Every other elementwise row answers one of two extremes — upload every launch, or upload once and never count it. Neither is what a caller has.
  * The real question is a ratio: data arrives, then some number of kernels run over it before it changes again. Below the break-even, re-sending per
  * launch is genuinely the right choice and residency is only bookkeeping; above it, residency is the whole game.
  *
  * Both rows carry the upload, so the comparison is honest at every point on the axis. The packed row is given **pre-interleaved** data — its best
  * case, since a caller with three separate arrays would have to pay for the interleave first — so the break-even this reports is conservative in
  * residency's disfavour.
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
class ResidencyBench:

  /** One upload of each array, then `launches` kernels over it. Divide the score by `launches` for the per-launch cost. */
  @Benchmark
  def resident(s: Residency, bh: Blackhole): Unit =
    s.resident.writeInputUnsafe("a", s.a)
    s.resident.writeInputUnsafe("b", s.b)
    s.resident.writeInputUnsafe("c", s.c)
    var k = 0
    while k < s.launches do
      s.resident.renderBatchIntoUnsafe(Map.empty, GeneratorBench.OneElement, s.native)
      k += 1
    bh.consume(s.native)

  /** The same work with the arrays re-sent every time, which is what encoding B has to do. */
  @Benchmark
  def perLaunchUpload(s: Residency, bh: Blackhole): Unit =
    var k = 0
    while k < s.launches do
      s.packed.renderBatchPackedUnsafe(Map.empty, s.packedData, Residency.Size, s.native)
      k += 1
    bh.consume(s.native)

object Residency:
  /** Fixed, because the axis under study is the launch count. Large enough that transfer dominates, small enough that 64 launches still fit an
    * iteration.
    */
  final val Size = 1000000

@State(Scope.Benchmark)
class Residency:

  @Param(Array("1", "2", "4", "8", "16", "64"))
  var launches: Int = 0

  var resident: Kernel[IO] = null
  var packed: Kernel[IO] = null
  var native: FloatBuffer = null
  var a: Array[Float] = null
  var b: Array[Float] = null
  var c: Array[Float] = null
  var packedData: Array[Float] = null

  private var releaseResident: IO[Unit] = IO.unit
  private var releasePacked: IO[Unit] = IO.unit

  @Setup(Level.Trial)
  def setup(): Unit =
    val (r, rr) = ClKernel.compile[IO](Formulas.elementwiseInputs, Residency.Size, 1).allocated.unsafeRunSync()
    val (p, rp) = ClKernel.compile[IO](Formulas.elementwise, 1, Residency.Size).allocated.unsafeRunSync()
    resident        = r
    packed          = p
    releaseResident = rr
    releasePacked   = rp
    native          = Data.directFloats(Residency.Size)
    a               = Data.fill(Residency.Size, 0x9e3779b9L)
    b               = Data.fill(Residency.Size, 0x517cc1b7L)
    c               = Data.fill(Residency.Size, 0x27220a95L)
    packedData      = new Array[Float](Residency.Size * 3)
    Data.interleave(a, b, c, packedData)

  @TearDown(Level.Trial)
  def teardown(): Unit =
    releasePacked.unsafeRunSync()
    releaseResident.unsafeRunSync()
    resident = null
    packed   = null
    native   = null
