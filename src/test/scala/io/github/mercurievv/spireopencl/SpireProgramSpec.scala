package io.github.mercurievv.spireopencl

import _root_.algebra.ring.Field
import cats.effect.IO

import io.github.mercurievv.spireopencl.opencl.ClKernel
import io.github.mercurievv.spireopencl.symbolic.{BinOp, Expr, Reify, UnOp, instances}
import spire.algebra.{NRoot, Trig}
import spire.implicits.*
import spire.math.{sin, sqrt}
import weaver.*

/** The promise on the tin: write a program in spire syntax, hand it arguments, get the answer off the GPU.
  *
  * Every program here is written once, against spire typeclasses only, and then used twice — at `Double`, where it is the arithmetic itself, and at
  * `Expr`, where it is the kernel. The two must agree; that they do is the whole claim.
  */
object SpireProgramSpec extends SimpleIOSuite:

  /** See KernelSpec: concurrent OpenCL context creation crashes the driver. */
  override def maxParallelism: Int = 1

  import instances.given

  /** Ordinary spire. No `Expr`, no import from this library, nothing about kernels. */
  private def program[V: {Field, Trig}](b: V, c: V, d: V): V = b * c - sin(d)

  private def compileAndRun(formula: symbolic.Formula, args: Map[String, Float]): IO[Float] =
    ClKernel.compile[IO](formula, size = 1, maxBatchSize = 1).use { kernel =>
      IO {
        val out = new Array[Float](1)
        kernel.renderUnsafe(Map.empty, args, out)
        out(0)
      }
    }

  test("a = b * c - sin(d): three arguments in, the answer out") {
    val formula = Reify(uniforms = Nil, params = List("b", "c", "d"))((_, arg) => program(arg("b"), arg("c"), arg("d")))
    val (b, c, d) = (2.5, 4.0, 0.75)
    val expected = program[Double](b, c, d)
    compileAndRun(formula, Map("b" -> b.toFloat, "c" -> c.toFloat, "d" -> d.toFloat)).map { actual =>
      expect(math.abs(actual.toDouble - expected) < 1e-5, f"kernel gave $actual%.6f, spire at Double gives $expected%.6f") &&
      // Not vacuous: the value must actually depend on all three arguments.
      expect(math.abs(expected - (b * c)) > 1e-3, "sin(d) contributed nothing to the reference")
    }
  }

  pureTest("the same program at Double and at Expr are the same program") {
    // Structural, not numeric: what the kernel computes is the tree the source built, with the arguments
    // still free. If reification silently substituted or reordered anything, this is where it shows.
    val formula = Reify(Nil, List("b", "c", "d"))((_, arg) => program(arg("b"), arg("c"), arg("d")))
    val expected = Expr.bin(
      BinOp.Add,
      Expr.mul(Expr.Param("b"), Expr.Param("c")),
      Expr.neg(Expr.un(UnOp.Sin, Expr.Param("d"))),
    )
    expect(formula.body == expected, s"reified as ${formula.body}")
  }

  test("the transcendental set spire asks for is real arithmetic, not a stub") {
    // Trig[Expr] promises exp/log/atan2/tanh/... and NRoot promises sqrt and fpow. Each must compile to
    // something the device computes correctly, or a caller's ordinary spire code silently produces nonsense.
    def wide[V: {Field, Trig, NRoot}](x: V, y: V): V =
      val t = summon[Trig[V]]
      t.exp(x) + t.log(y) + t.atan2(x, y) + t.tanh(x) + t.cosh(x) + sqrt(y) + t.expm1(x) + t.log1p(y) + t.acos(x) + t.pi
    val formula = Reify(Nil, List("x", "y"))((_, arg) => wide(arg("x"), arg("y")))
    val (x, y) = (0.4, 2.0)
    compileAndRun(formula, Map("x" -> x.toFloat, "y" -> y.toFloat)).map { actual =>
      val expected = wide[Double](x, y)
      expect(math.abs(actual.toDouble - expected) / math.abs(expected) < 1e-5, f"kernel $actual%.6f vs spire $expected%.6f")
    }
  }

  test("an integer power becomes multiplication, and still gives spire's answer") {
    def cubed[V: {Field, NRoot}](x: V): V = summon[NRoot[V]].fpow(x, summon[Field[V]].fromInt(3))
    val formula = Reify(Nil, List("x"))((_, arg) => cubed(arg("x")))
    // No pow() call in the emitted source: a small whole exponent is unrolled.
    val source = opencl.CodeGen(formula)
    compileAndRun(formula, Map("x" -> 1.5f)).map { actual =>
      expect(!source.contains("pow("), s"expected repeated multiplication, got:\n$source") &&
      expect(math.abs(actual - 3.375f) < 1e-5, s"got $actual")
    }
  }

  test("a fractional power still goes through pow") {
    def root[V: NRoot](x: V): V = summon[NRoot[V]].fpow(x, summon[NRoot[V]].sqrt(x))
    val formula = Reify(Nil, List("x"))((_, arg) => root(arg("x")))
    compileAndRun(formula, Map("x" -> 2.0f)).map { actual =>
      expect(math.abs(actual.toDouble - math.pow(2.0, math.sqrt(2.0))) < 1e-5, s"got $actual")
    }
  }

  pureTest("constants fold through the transcendentals, so a constant program compiles to a literal") {
    val formula = Reify(Nil, Nil)((_, _) => program[Expr](Expr.Const(2.0), Expr.Const(3.0), Expr.Const(0.0)))
    expect(formula.body == Expr.Const(6.0), s"folded to ${formula.body}")
  }
