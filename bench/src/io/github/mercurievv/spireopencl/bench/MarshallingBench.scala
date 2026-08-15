package io.github.mercurievv.spireopencl.bench

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.{ClKernel, Kernel}
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.util.concurrent.TimeUnit

/** Getting the data into the shape the library accepts.
  *
  * Every other benchmark starts from an `Array[Float]`, which is a convenient fiction. Data arrives as `Double` because that is what the JVM's
  * numeric code defaults to, or boxed because it came through a collection, or row-major because it came out of a columnar store as records. The
  * conversion is a full pass over the data and is invisible in every measurement above — but a caller pays it before the library is even called, so a
  * comparison that omits it is answering a question nobody asked.
  *
  * `direct` is the fiction, and the row everything else should be read against.
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
class MarshallingBench:

  /** The conversion alone, without the device, so it can be compared against a launch. */

  @Benchmark
  def narrowDouble(s: Marshalling, bh: Blackhole): Unit =
    var i = 0
    while i < s.size do
      s.floats(i) = s.doubles(i).toFloat
      i += 1
    bh.consume(s.floats)

  @Benchmark
  def unbox(s: Marshalling, bh: Blackhole): Unit =
    var i = 0
    while i < s.size do
      s.floats(i) = s.boxed(i)
      i += 1
    bh.consume(s.floats)

  /** A record-oriented source: one column pulled out of interleaved rows. The stride is what costs — every read touches a new cache line and the
    * prefetcher cannot help.
    */
  @Benchmark
  def gatherColumn(s: Marshalling, bh: Blackhole): Unit =
    var i = 0
    while i < s.size do
      s.floats(i) = s.rows(i * Marshalling.Columns)
      i += 1
    bh.consume(s.floats)

  /** For scale: the same number of floats copied with no conversion at all. */
  @Benchmark
  def copy(s: Marshalling, bh: Blackhole): Unit =
    System.arraycopy(s.source, 0, s.floats, 0, s.size)
    bh.consume(s.floats)

  /** The conversion plus the upload it exists to feed, which is what a caller starting from `Double` actually pays before a kernel can run. */
  @Benchmark
  def narrowAndUpload(s: Marshalling, bh: Blackhole): Unit =
    var i = 0
    while i < s.size do
      s.floats(i) = s.doubles(i).toFloat
      i += 1
    s.kernel.writeInputUnsafe("a", s.floats)
    bh.consume(s.floats)

  /** The upload on its own, so the pair above can be split. */
  @Benchmark
  def uploadOnly(s: Marshalling, bh: Blackhole): Unit =
    s.kernel.writeInputUnsafe("a", s.source)
    bh.consume(s.source)

object Marshalling:
  /** Fields per record in the row-major source, a plausible width for a struct pulled out of a columnar store. */
  final val Columns = 8

@State(Scope.Benchmark)
class Marshalling:

  @Param(Array("1000000", "10000000"))
  var size: Int = 0

  var doubles: Array[Double] = null
  var boxed: Array[java.lang.Float] = null
  var rows: Array[Float] = null
  var source: Array[Float] = null
  var floats: Array[Float] = null
  var kernel: Kernel[IO] = null

  private var release: IO[Unit] = IO.unit

  @Setup(Level.Trial)
  def setup(): Unit =
    source  = Data.fill(size, 0x9e3779b9L)
    doubles = Array.tabulate(size)(i => source(i).toDouble)
    boxed   = Array.tabulate(size)(i => java.lang.Float.valueOf(source(i)))
    rows    = new Array[Float](size * Marshalling.Columns)
    var i = 0
    while i < size do
      rows(i * Marshalling.Columns) = source(i)
      i += 1
    floats = new Array[Float](size)
    val (k, r) = ClKernel.compile[IO](Formulas.elementwiseInputs, size, 1).allocated.unsafeRunSync()
    kernel  = k
    release = r

  @TearDown(Level.Trial)
  def teardown(): Unit =
    release.unsafeRunSync()
    kernel = null
