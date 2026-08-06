package io.github.mercurievv.spireopencl

import _root_.algebra.ring.Field
import io.github.mercurievv.spireopencl.algebra.Trigonometry
import io.github.mercurievv.spireopencl.symbolic.{Expr, Reify, instances}
import weaver.*

/** The IR on its own: folding, sharing, and the reference interpreter that every kernel test measures against. */
object ExprSpec extends SimpleIOSuite:

  import instances.given

  private def noEnv(name: String): Double = throw new IllegalStateException(s"unexpected free name '$name'")

  pureTest("constant arithmetic folds away instead of reaching the kernel") {
    expect(Expr.add(Expr.Const(2), Expr.Const(3)) == Expr.Const(5.0)) &&
    expect(Expr.mul(Expr.Const(2), Expr.Const(3)) == Expr.Const(6.0)) &&
    expect(Expr.sin(Expr.Const(0)) == Expr.Const(0.0))
  }

  pureTest("identities drop the node rather than emitting a no-op multiply") {
    val x = Expr.Param("x")
    expect(Expr.add(x, Expr.zero) == x) &&
    expect(Expr.mul(x, Expr.one) == x) &&
    expect(Expr.mul(x, Expr.zero) == Expr.zero) &&
    expect(Expr.div(x, Expr.one) == x) &&
    expect(Expr.neg(Expr.neg(x)) == x)
  }

  pureTest("nodeCount counts the DAG, not the tree: a shared subexpression costs once") {
    val shared = Expr.sin(Expr.Param("x"))
    // shared + shared: Param, Un(Sin), Bin(Add) — three distinct nodes for four references.
    expect(Expr.nodeCount(Expr.add(shared, shared)) == 3)
  }

  pureTest("names reports free arguments in first-encountered order, and ignores the index") {
    val e = Expr.add(Expr.mul(Expr.Index, Expr.Uniform("dx")), Expr.Param("slope"))
    expect(Expr.names(e) == List("dx", "slope"))
  }

  pureTest("eval is the reference: the index and both argument kinds resolve") {
    val e = Expr.add(Expr.mul(Expr.Index, Expr.Uniform("dx")), Expr.Param("offset"))
    val env = Map("dx" -> 0.5, "offset" -> 2.0)
    expect(Expr.eval(env, index = 4.0)(e) == 4.0)
  }

  pureTest("comparison yields a number, so it can be multiplied in as a mask") {
    val gate = Expr.gt(Expr.Index, Expr.Uniform("threshold"))
    val env = Map("threshold" -> 3.0)
    expect(Expr.eval(env, index = 5.0)(gate) == 1.0) &&
    expect(Expr.eval(env, index = 1.0)(gate) == 0.0)
  }

  pureTest("evalSummed folds over the batch, and honours a scale applied above the sum") {
    val body = Expr.mul(Expr.Param("v"), Expr.Index)
    val envs = Seq(Map("v" -> 1.0), Map("v" -> 2.0), Map("v" -> 3.0))
    val summed = Expr.Sum(body)
    expect(Expr.evalSummed(envs, index = 2.0)(summed) == 12.0) &&
    expect(Expr.evalSummed(envs, index = 2.0)(Expr.mul(summed, Expr.Const(0.5))) == 6.0)
  }

  pureTest("eval refuses a Sum: a reduction has no meaning for a single element") {
    val thrown =
      try { val _ = Expr.eval(noEnv, 0.0)(Expr.Sum(Expr.one)); false }
      catch case _: IllegalArgumentException => true
    expect(thrown)
  }

  pureTest("polymorphic numeric code builds the tree it would have computed") {
    // The whole claim of the library: `shaped` knows nothing about Expr, yet instantiating it here yields the
    // expression, and instantiating it at Double yields the number that expression evaluates to.
    def shaped[V: {Field, Trigonometry}](x: V, gain: V): V =
      summon[Field[V]].times(x.sin, gain)

    import spire.std.double.given

    val symbolic = shaped[Expr](Expr.Param("x"), Expr.Const(0.5))
    val numeric = shaped[Double](0.7, 0.5)
    expect(math.abs(Expr.eval(Map("x" -> 0.7), 0.0)(symbolic) - numeric) < 1e-12)
  }

  pureTest("declared arguments are the binding order, and an undeclared one is refused") {
    val formula = Reify(List("dx"), List("slope")) { (uniform, param) =>
      Expr.mul(Expr.mul(Expr.Index, uniform("dx")), param("slope"))
    }
    val refused =
      try { val _ = Reify(List("dx"), Nil)((uniform, _) => uniform("nope")); false }
      catch case _: IllegalArgumentException => true
    expect(formula.uniforms == List("dx")) &&
    expect(formula.params == List("slope")) &&
    expect(refused)
  }

  pureTest("a folded-away argument keeps its declared slot") {
    // Why params are declared rather than discovered: `slope * 0` deletes the parameter from the tree, and a
    // derived argument list would shift every later kernel argument index under it.
    val formula = Reify(Nil, List("slope", "gain")) { (_, param) =>
      Expr.add(Expr.mul(param("slope"), Expr.zero), param("gain"))
    }
    expect(Expr.names(formula.body) == List("gain")) &&
    expect(formula.params == List("slope", "gain"))
  }

  pureTest("summed is idempotent and scaledBy lands above the reduction") {
    val formula = Reify(Nil, List("v"))((_, param) => param("v")).summed
    expect(formula.summed == formula) &&
    expect(formula.isReduced) &&
    expect(formula.scaledBy(2.0).body == Expr.mul(Expr.Sum(Expr.Param("v")), Expr.Const(2.0)))
  }
