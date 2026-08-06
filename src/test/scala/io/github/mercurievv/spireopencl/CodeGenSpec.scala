package io.github.mercurievv.spireopencl

import io.github.mercurievv.spireopencl.opencl.CodeGen
import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify}
import weaver.*

/** What the IR becomes, checked without a device. */
object CodeGenSpec extends SimpleIOSuite:

  private def rejects(f: => Any): Boolean =
    try { val _ = f; false }
    catch case _: IllegalArgumentException => true

  pureTest("a plain formula emits the element-major shape and reads its arguments") {
    val src = CodeGen(Reify(List("dx"), List("slope"))((u, p) => Expr.mul(Expr.mul(Expr.Index, u("dx")), p("slope"))))
    expect(src.contains("__kernel void compute(__global float* out, float dx, __global const float* params)")) &&
    expect(src.contains("int e = get_global_id(1);")) &&
    expect(src.contains("out[e * n + i]")) &&
    expect(src.contains("params[e]")) &&
    expect(!src.contains("barrier"))
  }

  pureTest("several parameters pack element-major, so the layout is independent of batch size") {
    val src = CodeGen(Reify(Nil, List("a", "b"))((_, p) => Expr.add(p("a"), p("b"))))
    expect(src.contains("params[e * 2 + 0]")) && expect(src.contains("params[e * 2 + 1]"))
  }

  pureTest("a reduced formula emits the barrier, the local accumulator and a batch-count argument") {
    val src = CodeGen(Reify(Nil, List("v"))((_, p) => p("v")).summed)
    expect(src.contains("int ne")) &&
    expect(src.contains("__local float* acc")) &&
    expect(src.contains("int e = get_local_id(1);")) &&
    expect(src.contains("barrier(CLK_LOCAL_MEM_FENCE);")) &&
    expect(src.contains("for (int k = 0; k < ne; k++) mix += acc[k];")) &&
    expect(src.contains("out[i] ="))
  }

  pureTest("work above the reduction is emitted once, after the barrier") {
    // The reason two-phase emission exists: a scale above the Sum must cost one multiply per work-item, not
    // one per batch element, so it has to land inside the `if (e == 0)` block.
    val src = CodeGen(Reify(Nil, List("v"))((_, p) => p("v")).summed.scaledBy(0.5))
    val afterBarrier = src.substring(src.indexOf("barrier"))
    expect(afterBarrier.contains("mix * 5.0000000000e-01")) &&
    expect(src.indexOf("5.0000000000e-01") > src.indexOf("barrier"))
  }

  pureTest("a shared subexpression is computed once on each side of the barrier, not per reference") {
    val shared = Expr.sin(Expr.Index)
    val src = CodeGen(Formula(Expr.add(shared, Expr.mul(shared, Expr.Const(2))), Nil, Nil))
    expect(src.linesIterator.count(_.contains("sin(fi)")) == 1)
  }

  pureTest("an argument name the kernel uses for itself is rejected") {
    expect(rejects(CodeGen(Formula(Expr.Param("out"), Nil, List("out"))))) &&
    expect(rejects(CodeGen(Formula(Expr.Uniform("acc"), List("acc"), Nil)))) &&
    expect(rejects(CodeGen(Formula(Expr.Index, List("i"), Nil))))
  }

  pureTest("a name declared as both a uniform and a parameter is rejected") {
    expect(rejects(CodeGen(Formula(Expr.Param("x"), List("x"), List("x")))))
  }

  pureTest("nested reductions are rejected: there is one batch dimension") {
    expect(rejects(CodeGen(Formula(Expr.Sum(Expr.Sum(Expr.Param("v"))), Nil, List("v"))))) &&
    // Two separate reductions are the same problem: only one can be the barrier.
    expect(rejects(CodeGen(Formula(Expr.add(Expr.Sum(Expr.Param("v")), Expr.Sum(Expr.Param("w"))), Nil, List("v", "w")))))
  }

  pureTest("work above the reduction may still be arbitrary algebra") {
    val src = CodeGen(Formula(Expr.sin(Expr.Sum(Expr.Param("v"))), Nil, List("v")))
    expect(src.substring(src.indexOf("barrier")).contains("sin(mix)"))
  }

  pureTest("a non-finite constant is refused rather than emitted as an unparseable literal") {
    expect(rejects(CodeGen(Formula(Expr.Const(Double.PositiveInfinity), Nil, Nil))))
  }
