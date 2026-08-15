package io.github.mercurievv.spireopencl.bench

import breeze.linalg.DenseVector
import cats.effect.IO
import io.github.mercurievv.spireopencl.opencl.ClKernel
import weaver.*

/** The gate every number depends on: the four contenders must compute the same thing.
  *
  * A benchmark that compares two different computations is the standard way this exercise goes wrong, and nothing about the timings reveals it — the
  * GPU row being four times faster looks exactly the same whether it is faster or whether it is computing a shorter expression. So this runs first
  * and the benchmarks are only worth reading if it passes.
  *
  * Sizes are small: correctness does not need 10^7, and the benchmark's own sizes would make this a slow test for no extra confidence.
  */
object AgreementSpec extends SimpleIOSuite:

  /** See `KernelSpec`: concurrent OpenCL context creation crashes pocl and, sometimes, the Apple driver. */
  override def maxParallelism: Int = 1

  private val n = 256

  /** Single precision, and the GPU's transcendentals are not bit-identical to `StrictMath`'s — OpenCL's `sin` is allowed 4 ULP and `exp` 3 ULP. A
    * relative tolerance is the only correct comparison; an exact one would fail for a reason that is not a bug.
    */
  private def close(a: Float, b: Float): Boolean =
    val d = math.abs(a - b)
    d <= 1e-4f * math.max(1.0f, math.max(math.abs(a), math.abs(b)))

  private def agree(name: String, expected: Array[Float], actual: Array[Float]): Expectations =
    val bad = (0 until expected.length).find(i => !close(expected(i), actual(i)))
    bad match
      case None    => success
      case Some(i) => failure(s"$name disagrees at $i: expected ${expected(i)}, got ${actual(i)}")

  test("encoding A, chain: kernel, spire, breeze and the plain loop all agree") {
    val expected = new Array[Float](n)
    Jvm.plainGeneratorChain(Bench.UniformA, Bench.UniformB, Bench.Depth, expected)

    val spire = new Array[Float](n)
    Jvm.spireGeneratorChain(Bench.UniformA, Bench.UniformB, Bench.Depth, spire)

    val breeze = Jvm.breezeGeneratorChain(Bench.UniformA, Bench.UniformB, Bench.Depth, n).toArray

    ClKernel.compile[IO](Formulas.generatorChain(Bench.Depth), n, 1).use { kernel =>
      IO {
        val gpu = Data.directFloats(n)
        kernel.renderBatchIntoUnsafe(
          Map("a" -> Bench.UniformA, "b" -> Bench.UniformB),
          GeneratorBench.OneElement,
          gpu,
          (_, _) => (),
        )
        val got = new Array[Float](n)
        gpu.duplicate().get(got)
        agree("spire", expected, spire) and agree("breeze", expected, breeze) and agree("opencl", expected, got)
      }
    }
  }

  test("encoding A, heavy: kernel, spire and the plain loop all agree") {
    val expected = new Array[Float](n)
    Jvm.plainGeneratorHeavy(Bench.UniformA, expected)

    val spire = new Array[Float](n)
    Jvm.spireGeneratorHeavy(Bench.UniformA, spire)

    ClKernel.compile[IO](Formulas.generatorHeavy, n, 1).use { kernel =>
      IO {
        val gpu = Data.directFloats(n)
        kernel.renderBatchIntoUnsafe(Map("a" -> Bench.UniformA), GeneratorBench.OneElement, gpu, (_, _) => ())
        val got = new Array[Float](n)
        gpu.duplicate().get(got)
        agree("spire", expected, spire) and agree("opencl", expected, got)
      }
    }
  }

  test("encoding B, a * b + c: kernel, spire, breeze and the plain loop all agree") {
    val a = Data.fill(n, 0x9e3779b9L)
    val b = Data.fill(n, 0x517cc1b7L)
    val c = Data.fill(n, 0x27220a95L)

    val expected = new Array[Float](n)
    Jvm.plainElementwise(a, b, c, expected)

    val spire = new Array[Float](n)
    Jvm.spireElementwise(a, b, c, spire)

    val breeze = Jvm.breezeElementwise(DenseVector(a), DenseVector(b), DenseVector(c)).toArray

    val packed = new Array[Float](n * 3)
    Data.interleave(a, b, c, packed)

    ClKernel.compile[IO](Formulas.elementwise, size = 1, maxBatchSize = n).use { kernel =>
      IO {
        val gpu = Data.directFloats(n)
        kernel.renderBatchPackedUnsafe(Map.empty, packed, n, gpu, (_, _) => ())
        val got = new Array[Float](n)
        gpu.duplicate().get(got)
        agree("spire", expected, spire) and agree("breeze", expected, breeze) and agree("opencl", expected, got)
      }
    }
  }

  test("encoding C, device-resident arrays: same answer as encoding B and as the JVM") {
    val a = Data.fill(n, 0x9e3779b9L)
    val b = Data.fill(n, 0x517cc1b7L)
    val c = Data.fill(n, 0x27220a95L)

    val expected = new Array[Float](n)
    Jvm.plainElementwise(a, b, c, expected)

    val heavyExpected = new Array[Float](n)
    Jvm.plainElementwiseHeavy(a, b, c, heavyExpected)

    ClKernel.compile[IO](Formulas.elementwiseInputs, size = n, maxBatchSize = 1).use { plain =>
      ClKernel.compile[IO](Formulas.elementwiseHeavyInputs, size = n, maxBatchSize = 1).use { heavy =>
        IO {
          val got = new Array[Float](n)
          val gotHeavy = new Array[Float](n)
          List(plain, heavy).foreach { k =>
            k.writeInputUnsafe("a", a)
            k.writeInputUnsafe("b", b)
            k.writeInputUnsafe("c", c)
          }
          plain.renderBatchUnsafe(Map.empty, GeneratorBench.OneElement, got)
          heavy.renderBatchUnsafe(Map.empty, GeneratorBench.OneElement, gotHeavy)
          agree("opencl input", expected, got) and agree("opencl input heavy", heavyExpected, gotHeavy)
        }
      }
    }
  }

  test("encoding B, heavy: kernel, spire, breeze and the plain loop all agree") {
    val a = Data.fill(n, 0x9e3779b9L)
    val b = Data.fill(n, 0x517cc1b7L)
    val c = Data.fill(n, 0x27220a95L)

    val expected = new Array[Float](n)
    Jvm.plainElementwiseHeavy(a, b, c, expected)

    val spire = new Array[Float](n)
    Jvm.spireElementwiseHeavy(a, b, c, spire)

    val breeze = Jvm.breezeElementwiseHeavy(DenseVector(a), DenseVector(b), DenseVector(c)).toArray

    val packed = new Array[Float](n * 3)
    Data.interleave(a, b, c, packed)

    ClKernel.compile[IO](Formulas.elementwiseHeavy, size = 1, maxBatchSize = n).use { kernel =>
      IO {
        val gpu = Data.directFloats(n)
        kernel.renderBatchPackedUnsafe(Map.empty, packed, n, gpu, (_, _) => ())
        val got = new Array[Float](n)
        gpu.duplicate().get(got)
        agree("spire", expected, spire) and agree("breeze", expected, breeze) and agree("opencl", expected, got)
      }
    }
  }
