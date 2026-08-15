package io.github.mercurievv.spireopencl.bench

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import io.github.mercurievv.spireopencl.opencl.ClKernel
import io.github.mercurievv.spireopencl.symbolic.{BinOp, Expr, Formula, Reify, UnOp, instances}
import instances.given

/** How wrong the device is, measured rather than assumed.
  *
  * Everything else in this module measures speed, which is only ever half the question: a kernel that is fifty times faster and quietly less accurate
  * is not fifty times better, and nothing in a timing reveals which one you have. OpenCL does not promise correctly rounded transcendentals — the
  * specification allows `sin` 4 ULP, `exp` 3, and `pow` **16** — so the error is not a bug to be found but a documented quantity to be checked
  * against.
  *
  * The reference is the same operation in `Double`, rounded to `Float`. `UnOp` carries its own `Double` definition, which is what the IR is
  * interpreted with and what `CodeGen` claims to compile faithfully, so the comparison is between the two halves of a claim the library already
  * makes.
  *
  * This also guards the library: nothing else here would notice a change to `CodeGen` that kept every answer plausible and lost three bits.
  */
object Precision:

  /** Distance in representable floats. Bit patterns are ordered so that adjacent floats differ by one, which makes "wrong by 2 ULP" a count rather
    * than a ratio — the only scale on which an error near zero and an error near 10^6 can be compared at all.
    */
  def ulps(a: Float, b: Float): Long =
    if a == b then 0L
    else if a.isNaN || b.isNaN then Long.MaxValue
    else
      def ordered(f: Float): Long =
        val bits = java.lang.Float.floatToIntBits(f).toLong
        if bits < 0 then 0x80000000L - bits else bits
      math.abs(ordered(a) - ordered(b))

  final case class Stats(
    op: String,
    samples: Int,
    max: Long,
    p99: Long,
    median: Long,
    worstInput: Float,
    expected: Float,
    actual: Float):
    def within(limit: Long): Boolean = max <= limit

  private def stats(op: String, inputs: Array[Float], expected: Array[Float], actual: Array[Float]): Stats =
    val errors = Array.tabulate(inputs.length)(i => ulps(expected(i), actual(i)))
    val sorted = errors.sorted
    val worst = errors.indices.maxBy(errors)
    Stats(
      op,
      inputs.length,
      sorted.last,
      sorted((sorted.length * 99) / 100),
      sorted(sorted.length / 2),
      inputs(worst),
      expected(worst),
      actual(worst),
    )

  /** Runs a one-input formula over the whole sample set in a single launch, using a device-resident input array — which is also the least contrived
    * use of `Expr.Input` there is.
    */
  private def onDevice(formula: Formula, samples: Array[Float]): Array[Float] =
    ClKernel
      .compile[IO](formula, size = samples.length, maxBatchSize = 1)
      .use { kernel =>
        IO {
          kernel.writeInputUnsafe("x", samples)
          val out = new Array[Float](samples.length)
          kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
          out
        }
      }
      .unsafeRunSync()

  /** Evenly spaced across the domain rather than random: a sweep visits the awkward parts of a range — the ends, and where an implementation switches
    * argument-reduction strategy — with the same certainty every time, and reproducibility is worth more here than coverage of the improbable.
    */
  def sweep(lo: Float, hi: Float, n: Int): Array[Float] =
    Array.tabulate(n)(i => (lo.toDouble + (hi.toDouble - lo.toDouble) * i / (n - 1)).toFloat)

  /** The domain each operation is measured over, chosen to be where the function is defined and where a caller would actually evaluate it. The wide
    * trigonometric range is deliberate: argument reduction is where an implementation loses accuracy, and a sweep over [-pi, pi] would never show it.
    */
  val domains: Map[UnOp, (Float, Float)] = Map(
    UnOp.Neg   -> (-1e6f, 1e6f),
    UnOp.Sin   -> (-1000f, 1000f),
    UnOp.Cos   -> (-1000f, 1000f),
    UnOp.Tan   -> (-100f, 100f),
    UnOp.Asin  -> (-1f, 1f),
    UnOp.Acos  -> (-1f, 1f),
    UnOp.Atan  -> (-1e3f, 1e3f),
    UnOp.Sinh  -> (-10f, 10f),
    UnOp.Cosh  -> (-10f, 10f),
    UnOp.Tanh  -> (-10f, 10f),
    UnOp.Exp   -> (-20f, 20f),
    UnOp.Expm1 -> (-20f, 20f),
    UnOp.Log   -> (1e-6f, 1e6f),
    UnOp.Log1p -> (-0.9f, 1e6f),
    UnOp.Sqrt  -> (0f, 1e6f),
  )

  /** What the OpenCL 1.2 full profile permits for single precision, in ULP. These are the numbers to hold the device to: exceeding one is a driver
    * defect or a code generation mistake, and staying far inside one is not a reason to relax the bound.
    */
  val allowed: Map[UnOp, Long] = Map(
    UnOp.Neg   -> 0L,
    UnOp.Sin   -> 4L,
    UnOp.Cos   -> 4L,
    UnOp.Tan   -> 5L,
    UnOp.Asin  -> 4L,
    UnOp.Acos  -> 4L,
    UnOp.Atan  -> 5L,
    UnOp.Sinh  -> 4L,
    UnOp.Cosh  -> 4L,
    UnOp.Tanh  -> 5L,
    UnOp.Exp   -> 3L,
    UnOp.Expm1 -> 3L,
    UnOp.Log   -> 3L,
    UnOp.Log1p -> 2L,
    UnOp.Sqrt  -> 3L,
  )

  def measure(op: UnOp, samples: Int = 1 << 14): Stats =
    val (lo, hi) = domains(op)
    val xs = sweep(lo, hi, samples)
    val expected = xs.map(x => op.eval(x.toDouble).toFloat)
    val actual = onDevice(Reify.arrays(Nil, Nil, List("x"))((_, _, in) => Expr.un(op, in("x"))), xs)
    stats(op.toString, xs, expected, actual)

  /** Error against arithmetic depth, the accuracy counterpart of `DepthSweepBench`.
    *
    * Rounding accumulates along a dependent chain exactly as time does, so the same axis that shows the device is not compute-bound also shows what
    * the answer costs. Reported together, the two say what a deeper formula buys and what it spends.
    */
  def chainError(depth: Int, samples: Int = 1 << 12): Stats =
    val xs = sweep(-10f, 10f, samples)
    val formula = Reify.arrays(List("a", "b"), Nil, List("x")) { (uniform, _, in) =>
      Programs.chain(in("x"), uniform("a"), uniform("b"), depth)
    }
    val expected = xs.map { x =>
      // In Double, so the reference carries the rounding the device cannot: the difference is the accumulation.
      var acc = x.toDouble
      var k = 0
      while k < depth do
        acc = acc * Bench.UniformA.toDouble + Bench.UniformB.toDouble
        k += 1
      acc.toFloat
    }
    val actual = ClKernel
      .compile[IO](formula, size = xs.length, maxBatchSize = 1)
      .use { kernel =>
        IO {
          kernel.writeInputUnsafe("x", xs)
          val out = new Array[Float](xs.length)
          kernel.renderBatchUnsafe(Map("a" -> Bench.UniformA, "b" -> Bench.UniformB), Seq(Map.empty), out)
          out
        }
      }
      .unsafeRunSync()
    stats(s"chain(depth=$depth)", xs, expected, actual)

  /** Whether the device keeps subnormal results or flushes them to zero.
    *
    * OpenCL permits flushing, and most GPUs do it. It is not a rounding error but a cliff — the difference between a very small number and nothing at
    * all — and it decides whether an algorithm that relies on gradual underflow can run here. A speed benchmark would never show it.
    */
  def flushesSubnormals: Boolean =
    val xs = Array(1.0e-40f, 5.0e-42f, java.lang.Float.MIN_VALUE, 1.0e-39f)
    val padded = xs ++ Array.fill(60)(1.0f)
    val formula = Reify.arrays(Nil, Nil, List("x", "one"))((_, _, in) => Expr.bin(BinOp.Mul, in("x"), in("one")))
    val out = ClKernel
      .compile[IO](formula, size = padded.length, maxBatchSize = 1)
      .use { kernel =>
        IO {
          kernel.writeInputUnsafe("x", padded)
          kernel.writeInputUnsafe("one", Array.fill(padded.length)(1.0f))
          val o = new Array[Float](padded.length)
          kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), o)
          o
        }
      }
      .unsafeRunSync()
    xs.indices.exists(i => xs(i) != 0.0f && out(i) == 0.0f)

  def main(args: Array[String]): Unit =
    println("ULP error of the device against the same operation in Double, rounded to Float.")
    println("`allowed` is what the OpenCL 1.2 full profile permits for single precision.")
    println()
    println("  op        samples    median      p99      max  allowed          worst input -> expected / actual")

    val results = UnOp.values.toList.filter(domains.contains).map(op => op -> measure(op))
    results.foreach { (op, s) =>
      val verdict = if s.within(allowed(op)) then " " else " OVER"
      println(
        f"  ${op.toString}%-9s ${s.samples}%7d ${s.median}%9d ${s.p99}%8d ${s.max}%8d ${allowed(op)}%8d$verdict%-5s " +
          f"${s.worstInput}%14g -> ${s.expected}%g / ${s.actual}%g",
      )
    }

    println()
    println("  error along a dependent chain, the accuracy counterpart of the depth sweep:")
    List(1, 8, 32, 128).foreach { d =>
      val s = chainError(d)
      println(f"    ${s.op}%-20s median ${s.median}%4d   p99 ${s.p99}%4d   max ${s.max}%4d")
    }

    println()
    println(if flushesSubnormals then "  subnormals are FLUSHED TO ZERO on this device" else "  subnormals survive on this device")

    val over = results.filter((op, s) => !s.within(allowed(op)))
    println()
    if over.isEmpty then println("every operation is within the OpenCL specification")
    else
      println(s"${over.length} operation(s) exceed the specification:")
      over.foreach((op, s) => println(s"  $op: ${s.max} ULP, allowed ${allowed(op)}"))
      sys.exit(1)
