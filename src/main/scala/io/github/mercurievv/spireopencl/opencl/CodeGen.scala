package io.github.mercurievv.spireopencl.opencl

import io.github.mercurievv.spireopencl.symbolic.{BinOp, Expr, Formula, UnOp}

/** IR → OpenCL C.
  *
  * Single precision throughout: Apple Silicon reports no `cl_khr_fp64` and a `double` kernel fails to build there (see `DeviceProbe`). Precision is a
  * property of the generated source, not of the IR — `Expr` holds doubles and any backend may narrow.
  *
  * The tree is a DAG — a large stack of terms shares subexpressions and every folded constant — so nodes are emitted once into SSA temporaries and
  * referenced by name afterwards. Emitting the tree as one nested expression would repeat shared work and produce a source file the driver struggles
  * to parse.
  */
object CodeGen:

  val kernelName = "compute"

  /** Names the emitted kernel uses for itself. A declared argument may not collide with one. */
  private val reserved = Set("i", "fi", "e", "n", "ne", "out", "params", "acc", "mix", kernelName)

  /** Deterministic C float literal. `toString` would emit `1.0E-4`, which is legal C, but fixed formatting keeps generated sources diffable. */
  private def literal(v: Double): String =
    if v.isNaN || v.isInfinite then throw new IllegalArgumentException(s"non-finite constant in formula: $v")
    else f"$v%.10ef"

  /** The transcendentals are all OpenCL built-ins with the same names and semantics as `java.lang.Math`, which is why the IR can carry the `Double`
    * version of each operation as its definition and still be compiled faithfully. Only these four need special syntax.
    */
  private def binText(op: BinOp, a: String, b: String): String = op match
    case BinOp.Add => s"$a + $b"
    case BinOp.Mul => s"$a * $b"
    case BinOp.Div => s"$a / $b"
    // OrderS masks are numbers, not booleans: they get multiplied into expressions.
    case BinOp.Gt    => s"($a > $b ? 1.0f : 0.0f)"
    case BinOp.Lt    => s"($a < $b ? 1.0f : 0.0f)"
    case BinOp.Pow   => s"pow($a, $b)"
    case BinOp.Atan2 => s"atan2($a, $b)"

  private def unText(op: UnOp, a: String): String = op match
    case UnOp.Neg   => s"-$a"
    case UnOp.Sin   => s"sin($a)"
    case UnOp.Cos   => s"cos($a)"
    case UnOp.Tan   => s"tan($a)"
    case UnOp.Asin  => s"asin($a)"
    case UnOp.Acos  => s"acos($a)"
    case UnOp.Atan  => s"atan($a)"
    case UnOp.Sinh  => s"sinh($a)"
    case UnOp.Cosh  => s"cosh($a)"
    case UnOp.Tanh  => s"tanh($a)"
    case UnOp.Exp   => s"exp($a)"
    case UnOp.Expm1 => s"expm1($a)"
    case UnOp.Log   => s"log($a)"
    case UnOp.Log1p => s"log1p($a)"
    case UnOp.Sqrt  => s"sqrt($a)"

  private final case class Emit(names: Map[Expr, String], lines: Vector[String], next: Int)

  /** Where a named parameter sits in the packed parameter buffer. Element-major, so the layout does not depend on how many elements a launch carries. */
  private def slotOf(formula: Formula, name: String): String =
    val idx = formula.params.indexOf(name)
    if formula.params.size == 1 then "params[e]" else s"params[e * ${formula.params.size} + $idx]"

  /** Post-order walk. Constants, arguments and the index are referenced inline; every computed node becomes one `float vN = …;`. */
  private def emit(e: Expr, st: Emit, paramRef: String => String): (Emit, String) =
    st.names.get(e) match
      case Some(name) => (st, name)
      case None =>
        e match
          case Expr.Const(v)   => (st, literal(v))
          case Expr.Uniform(n) => (st, n)
          // Per batch element, not per work-item in dimension 0: one launch covers every element, so a parameter is a read at this element's slot.
          case Expr.Param(n) => (st, paramRef(n))
          case Expr.Index    => (st, "fi")
          case Expr.Bin(op, l, r) =>
            val (st1, a) = emit(l, st, paramRef)
            val (st2, b) = emit(r, st1, paramRef)
            define(e, binText(op, a, b), st2)
          case Expr.Un(op, a) =>
            val (st1, x) = emit(a, st, paramRef)
            define(e, unText(op, x), st1)
          case Expr.Sum(_) => throw new IllegalArgumentException("Sum is only valid at the root of a formula")

  private def define(e: Expr, text: String, st: Emit): (Emit, String) =
    val name = s"v${st.next}"
    (Emit(st.names + (e -> name), st.lines :+ s"  float $name = $text;", st.next + 1), name)

  /** Two shapes, chosen by whether the formula is an `Expr.Sum`.
    *
    * **Per element** — 2D, dimension 0 the work-item, dimension 1 the batch element, output element-major at `out[e * n + i]`. The caller reduces.
    *
    * **Reduced** — the same 2D shape, but each work-group is one work-item's elements, which sum through local memory so only the finished total
    * returns to the host. That removes `ne - 1` host-side additions and all but one buffer conversion per launch. Keeping the batch dimension
    * parallel is the point: a serial accumulation loop was tried and cost 3128 us against 1012 us at 16 elements, because 128 work-items cannot fill
    * the device on their own.
    *
    * In both shapes one launch covers every element: at small sizes the per-launch latency, not the arithmetic, is the cost. All per-element
    * parameters share one packed buffer for the same reason — every extra host→device transfer costs about what the kernel does.
    */
  def apply(formula: Formula): String =
    (formula.uniforms ++ formula.params)
      .find(reserved.contains)
      .foreach(n => throw new IllegalArgumentException(s"argument name '$n' is reserved by the kernel"))
    formula.uniforms
      .find(formula.params.contains)
      .foreach(n => throw new IllegalArgumentException(s"'$n' is declared both as a uniform and as a parameter"))
    val paramRef = slotOf(formula, _)

    /* Emitted in two phases when the formula reduces, because the two halves run in different places: everything
     * below the Sum is per element and runs before the barrier, everything above it operates on the finished
     * total and runs once, in the reducing work-item. Seeding the second phase with the first phase's names --
     * including the sum itself, bound to `mix` -- is what lets a shared subexpression be computed once and used
     * on both sides. */
    val (perElement, elementResult, post, result) = formula.sum match
      case Some(inner) =>
        val (a, total) = emit(inner, Emit(Map.empty, Vector.empty, 0), paramRef)
        val seeded = a.copy(names = a.names + (Expr.Sum(inner) -> "mix"), lines = Vector.empty)
        val (b, out) = emit(formula.body, seeded, paramRef)
        (a.lines, total, b.lines, out)
      case None =>
        val (a, out) = emit(formula.body, Emit(Map.empty, Vector.empty, 0), paramRef)
        (a.lines, out, Vector.empty, out)

    val scalars = (formula.uniforms.map(n => s"float $n") ++ (if formula.isReduced then List("int ne") else Nil))
      .map(", " + _)
      .mkString
    val paramArgs = if formula.params.isEmpty then "" else ", __global const float* params"
    // One float per batch element, scoped to the work-group, which the host sizes at launch.
    val localArgs = if formula.isReduced then ", __local float* acc" else ""

    val preamble =
      s"""  int i = get_global_id(0);
         |  float fi = (float)i;""".stripMargin

    val core =
      if formula.isReduced then
        s"""  int e = get_local_id(1);
           |${perElement.mkString("\n")}
           |  acc[e] = $elementResult;
           |  barrier(CLK_LOCAL_MEM_FENCE);
           |  if (e == 0) {
           |    float mix = 0.0f;
           |    for (int k = 0; k < ne; k++) mix += acc[k];
           |${post.map("  " + _).mkString("\n")}
           |    out[i] = $result;
           |  }""".stripMargin
      else
        s"""  int e = get_global_id(1);
           |  int n = get_global_size(0);
           |${perElement.mkString("\n")}
           |  out[e * n + i] = $result;""".stripMargin

    s"""__kernel void $kernelName(__global float* out$scalars$paramArgs$localArgs) {
       |$preamble
       |$core
       |}
       |""".stripMargin
