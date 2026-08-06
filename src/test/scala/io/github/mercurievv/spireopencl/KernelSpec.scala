package io.github.mercurievv.spireopencl

import cats.effect.IO

import io.github.mercurievv.spireopencl.opencl.ClKernel
import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify}
import weaver.*

/** The compiled kernel against the reference interpreter.
  *
  * Tolerance is 1e-4 throughout: the kernel is single precision because Apple Silicon has no `cl_khr_fp64`.
  *
  * Requires an OpenCL device.
  */
object KernelSpec extends SimpleIOSuite:

  /** One test at a time. Weaver runs a suite's tests concurrently by default, and each of these builds and tears down its own OpenCL context —
    * concurrently creating and releasing contexts segfaults the driver, on pocl reliably and on Apple's occasionally. That is a property of the
    * drivers, not of this library, but a test suite that crashes the JVM reports nothing at all, so it is not something to leave to chance.
    */
  override def maxParallelism: Int = 1

  private val size = 128

  /** Something with every node kind in it: an index-derived ramp, a uniform, two parameters, a sine and a mask. */
  private val formula = Reify(List("t0", "dt"), List("freq", "gate")) { (uniform, param) =>
    val t = Expr.add(uniform("t0"), Expr.mul(Expr.Index, uniform("dt")))
    Expr.mul(Expr.sin(Expr.mul(t, param("freq"))), Expr.gt(t, param("gate")))
  }

  private val uniforms = Map("t0" -> 0.25f, "dt" -> 0.01f)

  private def reference(params: Map[String, Float], i: Int): Double =
    Expr.eval(name => params.getOrElse(name, uniforms(name)).toDouble, i.toDouble)(formula.body)

  private def worst(actual: Array[Float], offset: Int, expected: Int => Double): Double =
    (0 until size).map(i => math.abs(actual(offset + i).toDouble - expected(i))).max

  test("the kernel computes what the tree says it computes") {
    val params = Map("freq" -> 3.0f, "gate" -> 0.4f)
    ClKernel.compile[IO](formula, size, 1).use { kernel =>
      IO {
        val out = new Array[Float](size)
        kernel.renderUnsafe(uniforms, params, out)
        val diff = worst(out, 0, reference(params, _))
        // The mask must actually fire inside this window, or the test would pass on all-zeros.
        val gated = (0 until size).count(i => reference(params, i) == 0.0)
        expect(diff < 1e-4, f"kernel differs from Expr.eval by $diff%.3e") &&
        expect(gated > 0 && gated < size, s"the gate did not switch within the window ($gated of $size zero)")
      }
    }
  }

  test("one launch renders the whole batch, and each element lands in its own slice") {
    val batch = Seq(
      Map("freq" -> 1.0f, "gate" -> 0.0f),
      Map("freq" -> 5.0f, "gate" -> 0.5f),
      Map("freq" -> 9.0f, "gate" -> 1.0f),
    )
    ClKernel.compile[IO](formula, size, batch.size).use { kernel =>
      IO {
        val out = new Array[Float](size * batch.size)
        kernel.renderBatchUnsafe(uniforms, batch, out)
        val diffs = batch.zipWithIndex.map { case (p, e) => worst(out, e * size, reference(p, _)) }
        // Distinct frequencies: if the slices were aliased or the packed buffer misindexed, at least one would be wrong.
        expect(diffs.max < 1e-4, f"worst slice differs by ${diffs.max}%.3e (per slice: ${diffs.mkString(", ")})")
      }
    }
  }

  test("a reduced kernel returns the sum of the elements, computed in one launch") {
    val batch = Seq(Map("freq" -> 2.0f, "gate" -> 0.0f), Map("freq" -> 7.0f, "gate" -> 0.6f))
    ClKernel.compile[IO](formula.summed, size, batch.size).use { kernel =>
      IO {
        val out = new Array[Float](size)
        kernel.renderBatchUnsafe(uniforms, batch, out)
        val diff = worst(out, 0, i => batch.map(p => reference(p, i)).sum)
        val loudest = (0 until size).map(i => math.abs(batch.map(p => reference(p, i)).sum)).max
        expect(loudest > 1e-3, f"the reference sum is all zero ($loudest%.3e)") &&
        expect(diff < 1e-4, f"reduced kernel differs from the summed reference by $diff%.3e")
      }
    }
  }

  test("a scale above the reduction is applied once to the total, not once per element") {
    val batch = Seq(Map("freq" -> 2.0f, "gate" -> 0.0f), Map("freq" -> 7.0f, "gate" -> 0.0f))
    ClKernel.compile[IO](formula.summed.scaledBy(0.25), size, batch.size).use { kernel =>
      IO {
        val out = new Array[Float](size)
        kernel.renderBatchUnsafe(uniforms, batch, out)
        val diff = worst(out, 0, i => batch.map(p => reference(p, i)).sum * 0.25)
        expect(diff < 1e-4, f"scaled kernel differs from total * gain by $diff%.3e")
      }
    }
  }

  test("a formula with no parameters at all still compiles and runs") {
    val constant = Reify(List("dx"), Nil)((uniform, _) => Expr.mul(Expr.Index, uniform("dx")))
    ClKernel.compile[IO](constant, size, 1).use { kernel =>
      IO {
        val out = new Array[Float](size)
        kernel.renderUnsafe(Map("dx" -> 0.5f), Map.empty, out)
        expect(math.abs(out(10) - 5.0f) < 1e-4)
      }
    }
  }

  test("an empty batch is silence, not a launch") {
    ClKernel.compile[IO](formula.summed, size, 2).use { kernel =>
      IO {
        val out = Array.fill(size)(7.0f)
        kernel.renderBatchUnsafe(uniforms, Seq.empty, out)
        expect(out.forall(_ == 0.0f))
      }
    }
  }

  test("a batch beyond what the kernel was compiled for is refused") {
    ClKernel.compile[IO](formula, size, 1).use { kernel =>
      IO(kernel.renderBatchUnsafe(uniforms, Seq(Map("freq" -> 1f, "gate" -> 0f), Map("freq" -> 2f, "gate" -> 0f)), new Array[Float](size * 2)))
        .as(failure("should have refused the oversized batch"))
        .handleError(e => expect(e.getMessage.contains("exceed the compiled maximum"), s"wrong failure: ${e.getMessage}"))
    }
  }

  test("a missing argument names itself rather than reading whatever was in the buffer") {
    ClKernel.compile[IO](formula, size, 1).use { kernel =>
      IO(kernel.renderUnsafe(uniforms, Map("freq" -> 1f), new Array[Float](size)))
        .as(failure("should have refused the incomplete parameter map"))
        .handleError(e => expect(e.getMessage.contains("gate"), s"wrong failure: ${e.getMessage}"))
    }
  }

  test("a batch the device cannot form a work-group for is rejected at compile, not at launch") {
    // A reduced kernel makes one work-group per work-item with one work-item per batch element, so the
    // device's work-group ceiling caps the batch. Failing here is the point: the alternative is a failed
    // launch in the middle of whatever the caller was doing.
    ClKernel.defaultDevice[IO].flatMap { dev =>
      ClKernel
        .compileOn[IO](dev, formula.summed, size, 100_000)
        .use(_ => IO.pure(failure("compile should have rejected the batch size")))
        .handleError(e => expect(e.getMessage.contains("work-group"), s"wrong failure: ${e.getMessage}"))
    }
  }

  test("a maximum batch of zero is rejected rather than producing a kernel that cannot run") {
    ClKernel.defaultDevice[IO].flatMap { dev =>
      ClKernel
        .compileOn[IO](dev, formula.summed, size, 0)
        .use(_ => IO.pure(failure("compile should have rejected a zero batch")))
        .handleError(e => expect(e.getMessage.contains("at least 1"), s"wrong failure: ${e.getMessage}"))
    }
  }

  test("a formula the driver cannot build reports the build log, not just a status code") {
    // Nothing in the IR can produce bad C, so this goes in through the one door that bypasses it: a
    // hand-made Formula whose emitted source is invalid.
    val broken = Formula(Expr.Uniform("undeclared_in_source"), Nil, Nil)
    ClKernel.defaultDevice[IO].flatMap { dev =>
      ClKernel
        .compileOn[IO](dev, broken, size, 1)
        .use(_ => IO.pure(failure("compile should have failed to build")))
        .handleError(e => expect(e.getMessage.contains("kernel build failed"), s"wrong failure: ${e.getMessage}"))
    }
  }
