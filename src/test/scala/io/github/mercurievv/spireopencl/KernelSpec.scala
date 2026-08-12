package io.github.mercurievv.spireopencl

import cats.Id
import cats.data.StateT
import cats.effect.IO

import io.github.mercurievv.spireopencl.opencl.ClKernel
import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify, instances}
import io.github.mercurievv.spireopencl.symbolic.state.{Store, Var, at}
import weaver.*

import instances.given

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
        expect(
          gated > 0 && gated < size,
          s"the gate did not switch within the window ($gated of $size zero)",
        )
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
        expect(
          diffs.max < 1e-4,
          f"worst slice differs by ${diffs.max}%.3e (per slice: ${diffs.mkString(", ")})",
        )
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
      IO(
        kernel.renderBatchUnsafe(
          uniforms,
          Seq(Map("freq" -> 1f, "gate" -> 0f), Map("freq" -> 2f, "gate" -> 0f)),
          new Array[Float](size * 2),
        ),
      )
        .as(failure("should have refused the oversized batch"))
        .handleError(e =>
          expect(
            e.getMessage.contains("exceed the compiled maximum"),
            s"wrong failure: ${e.getMessage}",
          ),
        )
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

  /** Reads its own cell and advances it by a per-element step, so a launch's output *is* what the previous launch left. */
  private val counter =
    Reify.stateful(Nil, List("step"), List("count"))((_, p, s) => (s("count"), Map("count" -> Expr.add(s("count"), p("step")))))

  private val steps = Seq(Map("step" -> 1.0f), Map("step" -> 10.0f))

  test("a cell survives to the next launch, and each batch element keeps its own") {
    ClKernel.compile[IO](counter, size, steps.size).use { kernel =>
      IO {
        val out = new Array[Float](size * steps.size)
        def launch(): (Float, Float) =
          kernel.renderBatchUnsafe(Map.empty, steps, out)
          (out(0), out(size))

        val first = launch()
        val second = launch()
        val third = launch()
        val cells = new Array[Float](steps.size)
        kernel.readStateUnsafe(cells)
        expect(first == ((0.0f, 0.0f)), s"cells start zeroed, got $first") &&
        expect(second == ((1.0f, 10.0f)), s"one launch's update, got $second") &&
        expect(third == ((2.0f, 20.0f)), s"updates accumulate, got $third") &&
        expect(
          cells.toList == List(3.0f, 30.0f),
          s"the write-back ran once more than was read back, got ${cells.toList}",
        ) &&
        // The cell is read by every work-item in dimension 0, not only the one that writes it back.
        expect(
          (0 until size).forall(i => out(i) == 2.0f),
          "some work-items read a different value than others",
        )
      }
    }
  }

  test("a cell on the device can be set, modified, and got back directly — no inference from output") {
    ClKernel.compile[IO](counter, size, steps.size).use { kernel =>
      IO {
        kernel.writeStateUnsafe(Array(100.0f, 200.0f))
        val seeded = new Array[Float](steps.size)
        kernel.readStateUnsafe(seeded)

        val out = new Array[Float](size * steps.size)
        kernel.renderBatchUnsafe(Map.empty, steps, out)
        val modified = new Array[Float](steps.size)
        kernel.readStateUnsafe(modified)

        expect(
          seeded.toList == List(100.0f, 200.0f),
          s"set did not stick before any launch, got ${seeded.toList}",
        ) &&
        expect(
          out(0) == 100.0f && out(size) == 200.0f,
          s"launch did not read the seeded value, got (${out(0)}, ${out(size)})",
        ) &&
        expect(
          modified.toList == List(101.0f, 210.0f),
          s"launch's write did not land on top of the seeded value, got ${modified.toList}",
        )
      }
    }
  }

  /** The same `counter` formula, but its body and update come out of the `Var`/`Store`/`At` layer ([[VarSpec]]'s last test) instead of being written
    * out by hand — a cell's step zoomed to id 0 in a `Store[Expr]`, seeded with the read placeholder `Expr.State("count")` rather than a real value,
    * since composing the tree happens before there is any value to seed with.
    */
  private val counterViaVar: Formula =
    val advance: Var[Id, Expr, Expr] = StateT(prev => (Expr.add(prev, Expr.Param("step")), prev))
    val (nextStore, output) = advance.at[Store[Expr]](0).run(Map(0 -> Expr.State("count")))
    Formula(
      output,
      uniforms = Nil,
      params   = List("step"),
      states   = List("count"),
      updates  = Map("count" -> nextStore(0)),
    )

  test("a cell built through Var.at — not written out by hand — can be set, modified, and got back on the device") {
    ClKernel.compile[IO](counterViaVar, size, steps.size).use { kernel =>
      IO {
        kernel.writeStateUnsafe(Array(100.0f, 200.0f))
        val seeded = new Array[Float](steps.size)
        kernel.readStateUnsafe(seeded)

        val out = new Array[Float](size * steps.size)
        kernel.renderBatchUnsafe(Map.empty, steps, out)
        val modified = new Array[Float](steps.size)
        kernel.readStateUnsafe(modified)

        expect(
          seeded.toList == List(100.0f, 200.0f),
          s"the Var-built formula did not read back what was set, got ${seeded.toList}",
        ) &&
        expect(
          out(0) == 100.0f && out(size) == 200.0f,
          s"the launch did not read the seeded value through Var.at's zoom, got (${out(0)}, ${out(size)})",
        ) &&
        expect(
          modified.toList == List(101.0f, 210.0f),
          s"Var.at's update did not land on the device, got ${modified.toList}",
        )
      }
    }
  }

  test("state works under the reduction too, where the cell is read before the barrier") {
    ClKernel.compile[IO](counter.summed, size, steps.size).use { kernel =>
      IO {
        val out = new Array[Float](size)
        def launch(): Float =
          kernel.renderBatchUnsafe(Map.empty, steps, out)
          out(0)

        val first = launch()
        val second = launch()
        expect(first == 0.0f, s"got $first") && expect(second == 11.0f, s"both elements advanced, got $second")
      }
    }
  }

  test("a shrinking batch does not disturb the cells of the elements that remain") {
    ClKernel.compile[IO](counter, size, steps.size).use { kernel =>
      IO {
        val out = new Array[Float](size * steps.size)
        kernel.renderBatchUnsafe(Map.empty, steps, out)
        kernel.renderBatchUnsafe(Map.empty, steps.take(1), out)
        kernel.renderBatchUnsafe(Map.empty, steps, out)
        // Element 0 advanced through all three launches; element 1's slot was untouched by the short one.
        expect(out(0) == 2.0f, s"element 0 got ${out(0)}") && expect(out(size) == 10.0f, s"element 1 got ${out(size)}")
      }
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
        .handleError(e =>
          expect(
            e.getMessage.contains("kernel build failed"),
            s"wrong failure: ${e.getMessage}",
          ),
        )
    }
  }

  test("reading into native memory gives the same floats as reading into an array") {
    // The two entry points share a launch and differ only in the pointer handed to `clEnqueueReadBuffer`, so
    // anything but bit-identical output means the destination changed the arithmetic — which it must not.
    val batch = Seq(Map("freq" -> 3.0f, "gate" -> 0.4f), Map("freq" -> 5.0f, "gate" -> 0.1f))
    ClKernel.compile[IO](formula, size, batch.size).use { kernel =>
      IO {
        val viaArray = new Array[Float](size * batch.size)
        kernel.renderBatchUnsafe(uniforms, batch, viaArray)

        // Deliberately offset: the buffer's position is where the results must land, and a launch that ignored it
        // would still pass a test that always wrote at zero.
        val pad = 7
        val buf = java.nio.ByteBuffer
          .allocateDirect((size * batch.size + pad) * java.lang.Float.BYTES)
          .order(java.nio.ByteOrder.nativeOrder)
          .asFloatBuffer
        buf.position(pad)
        kernel.renderBatchIntoUnsafe(uniforms, batch, buf)

        val viaBuffer = Array.tabulate(size * batch.size)(i => buf.get(pad + i))
        expect(
          viaBuffer.sameElements(viaArray),
          "native readback differs from the array readback",
        ) &&
        expect(buf.position() == pad, "the launch moved the buffer's position") &&
        expect(
          viaArray.exists(_ != 0.0f),
          "the formula produced silence, so the comparison proves nothing",
        )
      }
    }
  }

  test("a heap buffer is refused rather than passed to the driver") {
    ClKernel.compile[IO](formula, size, 1).use { kernel =>
      IO {
        val heap = java.nio.FloatBuffer.allocate(size)
        val err = scala.util.Try(kernel.renderBatchIntoUnsafe(uniforms, Seq(Map("freq" -> 1.0f, "gate" -> 0.0f)), heap))
        expect(err.isFailure) &&
        expect(
          err.failed.get.getMessage.contains("direct"),
          s"wrong failure: ${err.failed.get.getMessage}",
        )
      }
    }
  }

  test("a packed batch launches to the same floats as a batch of maps") {
    // The packed entry point exists to remove a per-element Map from an audio path, so the one thing it must not do is
    // compute anything different. Bit-identical: same launch, same staging bytes, only the way they were named differs.
    val batch = Seq(Map("freq" -> 3.0f, "gate" -> 0.4f), Map("freq" -> 5.0f, "gate" -> 0.1f))
    ClKernel.compile[IO](formula, size, batch.size).use { kernel =>
      IO {
        val viaArray = new Array[Float](size * batch.size)
        kernel.renderBatchUnsafe(uniforms, batch, viaArray)

        def direct(n: Int) = java.nio.ByteBuffer
          .allocateDirect(n * java.lang.Float.BYTES)
          .order(java.nio.ByteOrder.nativeOrder)
          .asFloatBuffer

        // Element-major, in the order the formula declares its parameters -- "freq" then "gate".
        val packed = Array(3.0f, 0.4f, 5.0f, 0.1f)
        val buf = direct(size * batch.size)
        kernel.renderBatchPackedUnsafe(uniforms, packed, batch.size, buf)
        val viaPacked = Array.tabulate(size * batch.size)(buf.get)

        expect(viaPacked.sameElements(viaArray), "packed batch differs from the map batch") &&
        expect(
          viaArray.exists(_ != 0.0f),
          "the formula produced silence, so the comparison proves nothing",
        )
      }
    }
  }

  test("a packed batch too short for its element count is refused") {
    ClKernel.compile[IO](formula, size, 2).use { kernel =>
      IO {
        val buf = java.nio.ByteBuffer.allocateDirect(size * 2 * 4).order(java.nio.ByteOrder.nativeOrder).asFloatBuffer
        val err = scala.util.Try(kernel.renderBatchPackedUnsafe(uniforms, Array(1.0f, 2.0f), 2, buf))
        expect(err.isFailure) && expect(
          err.failed.get.getMessage.contains("need 4"),
          s"wrong failure: ${err.failed.get.getMessage}",
        )
      }
    }
  }
