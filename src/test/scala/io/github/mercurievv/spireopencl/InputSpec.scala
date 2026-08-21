package io.github.mercurievv.spireopencl

import _root_.algebra.ring.Field
import cats.effect.IO
import io.github.mercurievv.spireopencl.opencl.{ClKernel, CodeGen}
import io.github.mercurievv.spireopencl.symbolic.{Expr, Reify, instances}
import spire.implicits.*
import weaver.*

/** Device-resident input arrays: one value per work-item, uploaded once, read by every launch afterwards.
  *
  * The contract that matters is the separation — writing an array and launching against it are different calls, so a caller whose data does not
  * change between launches transfers it once instead of once per launch. Everything else here guards the ways that separation can go wrong.
  */
object InputSpec extends SimpleIOSuite:

  /** See KernelSpec: concurrent OpenCL context creation crashes the driver. */
  override def maxParallelism: Int = 1

  import instances.given

  private def rejects(f: => Any): Boolean =
    try { val _ = f; false }
    catch
      case _: IllegalArgumentException => true
      case _: IllegalStateException    => true

  /** Ordinary spire, polymorphic in its value type — the same source that would run on `Float`. */
  private def program[V: Field](a: V, b: V, c: V): V = a * b + c

  private val formula = Reify.arrays(Nil, Nil, List("a", "b", "c"))((_, _, in) => program(in("a"), in("b"), in("c")))

  private val n = 64

  private def data(seed: Int): Array[Float] = Array.tabulate(n)(i => (i * seed % 17).toFloat - 8.0f)

  private object Data:

    /** JOCL cannot take a pointer into the JVM heap, so a readback destination has to be allocated direct. */
    def directFloats(count: Int): java.nio.FloatBuffer =
      java.nio.ByteBuffer.allocateDirect(count * java.lang.Float.BYTES).order(java.nio.ByteOrder.nativeOrder).asFloatBuffer

  pureTest("each input becomes its own pointer argument, read at the work-item index") {
    val src = CodeGen(formula)
    expect(src.contains("__global const float* a, __global const float* b, __global const float* c")) &&
    expect(src.contains("= a[i];")) &&
    expect(src.contains("out[e * n + i]"))
  }

  pureTest("an input read twice is read from memory once") {
    val src = CodeGen(Reify.arrays(Nil, Nil, List("a"))((_, _, in) => Expr.mul(in("a"), in("a"))))
    expect(src.linesIterator.count(_.contains("a[i]")) == 1)
  }

  pureTest("an input may not collide with a uniform or a parameter") {
    expect(rejects(CodeGen(Reify.arrays(List("a"), Nil, List("a"))((u, _, _) => u("a"))))) &&
    expect(rejects(CodeGen(Reify.arrays(Nil, List("a"), List("a"))((_, p, _) => p("a")))))
  }

  pureTest("an input may not appear in a state update, for the same reason the index may not") {
    // Both vary per work-item, and a state cell has one slot per batch element — there is no work-item whose
    // answer would be the one to write.
    expect(
      rejects(
        Reify.stateful(Nil, Nil, List("s"))((_, _, s) => (s("s"), Map("s" -> Expr.add(s("s"), Expr.Input("a"))))),
      ),
    )
  }

  test("writeInputsT splits one case class of arrays into per-name writes, typed to the formula it built") {
    import io.github.mercurievv.spireopencl.opencl.writeInputsT
    case class Abc[V](a: V, b: V, c: V)
    val typedFormula = Reify.arraysTyped[Abc](Nil, Nil)((_, _, abc) => program(abc.a, abc.b, abc.c))
    val (a, b, c) = (data(3), data(5), data(7))
    val expected = Array.tabulate(n)(i => program[Float](a(i), b(i), c(i)))
    ClKernel.compileT[IO, Abc](typedFormula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        kernel.writeInputsT(Abc(a, b, c))
        val out = new Array[Float](n)
        kernel.kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
        expect(out.toList == expected.toList)
      }
    }
  }

  test("a * b + c over three arrays gives the same floats as the same program on the JVM") {
    val (a, b, c) = (data(3), data(5), data(7))
    val expected = Array.tabulate(n)(i => program[Float](a(i), b(i), c(i)))
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        kernel.writeInputUnsafe("b", b)
        kernel.writeInputUnsafe("c", c)
        val out = new Array[Float](n)
        kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
        expect(out.toList == expected.toList)
      }
    }
  }

  test("an array written once is still there for the next launch") {
    // The whole point: the second launch transfers nothing and must answer identically.
    val (a, b, c) = (data(3), data(5), data(7))
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        kernel.writeInputUnsafe("b", b)
        kernel.writeInputUnsafe("c", c)
        val first = new Array[Float](n)
        val second = new Array[Float](n)
        kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), first)
        kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), second)
        expect(first.toList == second.toList)
      }
    }
  }

  test("rewriting one array changes the answer and leaves the others alone") {
    val (a, b, c) = (data(3), data(5), data(7))
    val a2 = data(11)
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        kernel.writeInputUnsafe("b", b)
        kernel.writeInputUnsafe("c", c)
        val out = new Array[Float](n)
        kernel.writeInputUnsafe("a", a2)
        kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
        expect(out.toList == Array.tabulate(n)(i => program[Float](a2(i), b(i), c(i))).toList)
      }
    }
  }

  test("launching before an array is written is refused rather than answered with zeros") {
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", data(3))
        expect(rejects(kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), new Array[Float](n))))
      }
    }
  }

  test("an array shorter than the compiled size is refused, and an undeclared name too") {
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        expect(rejects(kernel.writeInputUnsafe("a", new Array[Float](n - 1)))) &&
        expect(rejects(kernel.writeInputUnsafe("d", new Array[Float](n)))) &&
        expect(rejects(kernel.writeInputFromUnsafe("a", java.nio.FloatBuffer.allocate(n))))
      }
    }
  }

  test("native memory works as a source, and reports the compiled size") {
    val a = data(3)
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        val direct = java.nio.ByteBuffer
          .allocateDirect(n * java.lang.Float.BYTES)
          .order(java.nio.ByteOrder.nativeOrder)
          .asFloatBuffer
        direct.put(a)
        direct.position(0)
        kernel.writeInputFromUnsafe("a", direct)
        kernel.writeInputUnsafe("b", data(5))
        kernel.writeInputUnsafe("c", data(7))
        val out = new Array[Float](n)
        kernel.renderBatchUnsafe(Map.empty, Seq(Map.empty), out)
        expect(kernel.size == n) &&
        expect(direct.position() == 0) &&
        expect(out.toList == Array.tabulate(n)(i => program[Float](a(i), data(5)(i), data(7)(i))).toList)
      }
    }
  }

  test("a mapped readback gives the same floats as a copied one") {
    val (a, b, c) = (data(3), data(5), data(7))
    val expected = Array.tabulate(n)(i => program[Float](a(i), b(i), c(i)))
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1, hostVisibleOutput = true).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        kernel.writeInputUnsafe("b", b)
        kernel.writeInputUnsafe("c", c)
        val got = kernel.renderBatchMappedUnsafe(Map.empty, Seq(Map.empty)) { mapped =>
          val copy = new Array[Float](mapped.remaining)
          mapped.get(copy)
          copy.toList
        }
        expect(got == expected.toList)
      }
    }
  }

  test("mapping is refused on a kernel not compiled for it, rather than silently copying") {
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", data(3))
        kernel.writeInputUnsafe("b", data(5))
        kernel.writeInputUnsafe("c", data(7))
        expect(rejects(kernel.renderBatchMappedUnsafe(Map.empty, Seq(Map.empty))(_.remaining)))
      }
    }
  }

  test("launches in flight land in separate slots and all survive to be collected") {
    // The contract that makes pipelining safe: several enqueued launches must not overwrite each other,
    // and each slot must still hold its own answer once the queue drains.
    val (a, b, c) = (data(3), data(5), data(7))
    val scaled = Reify.arrays(Nil, List("g"), List("a", "b", "c"))((_, p, in) => Expr.mul(program(in("a"), in("b"), in("c")), p("g")))
    ClKernel.compile[IO](scaled, size = n, maxBatchSize = 1, outputSlots = 4).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        kernel.writeInputUnsafe("b", b)
        kernel.writeInputUnsafe("c", c)
        val gains = List(1.0f, 2.0f, 3.0f, 4.0f)
        val slots = gains.map(g => kernel.enqueueBatchUnsafe(Map.empty, Seq(Map("g" -> g))))
        kernel.finishUnsafe()
        val got = slots.map { slot =>
          val buf = Data.directFloats(n)
          kernel.readSlotUnsafe(slot, buf)
          val out = new Array[Float](n)
          buf.duplicate().get(out)
          out.toList
        }
        val expected = gains.map(g => Array.tabulate(n)(i => program[Float](a(i), b(i), c(i)) * g).toList)
        expect(slots == List(0, 1, 2, 3)) and expect(kernel.outputSlots == 4) and expect(got == expected)
      }
    }
  }

  test("enqueued readbacks land correctly once the queue has drained") {
    // The pipelined shape end to end: nothing waits until `finishUnsafe`, and every destination must then
    // hold its own launch's answer. Each gets its own buffer because an enqueued read writes whenever the
    // driver reaches it.
    val (a, b, c) = (data(3), data(5), data(7))
    val scaled = Reify.arrays(Nil, List("g"), List("a", "b", "c"))((_, p, in) => Expr.mul(program(in("a"), in("b"), in("c")), p("g")))
    ClKernel.compile[IO](scaled, size = n, maxBatchSize = 1, outputSlots = 4).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        kernel.writeInputUnsafe("b", b)
        kernel.writeInputUnsafe("c", c)
        val gains = List(1.0f, 2.0f, 3.0f, 4.0f)
        val destinations = gains.map(_ => Data.directFloats(n))
        gains.zip(destinations).foreach { (g, dest) =>
          val slot = kernel.enqueueBatchUnsafe(Map.empty, Seq(Map("g" -> g)))
          kernel.enqueueReadSlotUnsafe(slot, dest)
        }
        kernel.finishUnsafe()
        val got = destinations.map { buf =>
          val out = new Array[Float](n)
          buf.duplicate().get(out)
          out.toList
        }
        expect(got == gains.map(g => Array.tabulate(n)(i => program[Float](a(i), b(i), c(i)) * g).toList))
      }
    }
  }

  test("slots wrap, and a slot outside the compiled range is refused") {
    ClKernel.compile[IO](formula, size = n, maxBatchSize = 1, outputSlots = 2).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", data(3))
        kernel.writeInputUnsafe("b", data(5))
        kernel.writeInputUnsafe("c", data(7))
        val slots = (0 until 5).map(_ => kernel.enqueueBatchUnsafe(Map.empty, Seq(Map.empty))).toList
        kernel.finishUnsafe()
        expect(slots == List(0, 1, 0, 1, 0)) and
          expect(rejects(kernel.readSlotUnsafe(2, Data.directFloats(n))))
      }
    }
  }

  test("inputs and per-element parameters coexist, each on its own dimension") {
    // An input varies along dimension 0, a parameter along dimension 1: one launch of two batch elements over one
    // array must scale the same array by two different numbers.
    val a = data(3)
    val scaled = Reify.arrays(Nil, List("g"), List("a"))((_, p, in) => Expr.mul(in("a"), p("g")))
    ClKernel.compile[IO](scaled, size = n, maxBatchSize = 2).use { kernel =>
      IO {
        kernel.writeInputUnsafe("a", a)
        val out = new Array[Float](n * 2)
        kernel.renderBatchUnsafe(Map.empty, Seq(Map("g" -> 2.0f), Map("g" -> -1.0f)), out)
        expect(out.take(n).toList == a.map(_ * 2.0f).toList) &&
        expect(out.drop(n).toList == a.map(_ * -1.0f).toList)
      }
    }
  }
