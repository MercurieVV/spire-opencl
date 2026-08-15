package io.github.mercurievv.spireopencl.bench

import io.github.mercurievv.spireopencl.symbolic.Expr

/** What each benchmark actually moves and computes per element, so a duration can be turned into something interpretable.
  *
  * "1923 us" says nothing on its own. The same run is 40 MB of output at 20.9 GB/s, which is a number you can hold against what the hardware can do —
  * and that comparison, not the duration, is what says whether there is anything left to win. None of it is in the JMH report: JMH times a method and
  * has no idea what the method touched, so the shapes have to be stated here, once, next to the benchmarks that have them.
  */
object Traffic:

  /** Per element of the array, in bytes and operations.
    *
    * `deviceRead`/`deviceWrite` is traffic against whichever memory the arithmetic runs out of — device DRAM for the kernels, host DRAM for the JVM
    * rows — which is what arithmetic intensity and the roofline are about, and is what makes the two sides comparable at all.
    *
    * `hostUp`/`hostDown` is what crosses the API boundary per launch. On a discrete GPU that is the PCIe traffic; on Apple Silicon the two memories
    * are one, so this is a copy rather than a bus transfer — still real, still the thing residency and mapping remove.
    */
  final case class Profile(
    deviceRead: Int,
    deviceWrite: Int,
    hostUp: Int,
    hostDown: Int,
    ops: Int):
    def deviceBytes: Int = deviceRead + deviceWrite
    def hostBytes: Int = hostUp + hostDown

    /** Operations per byte of memory traffic — where a kernel sits on the roofline. Low means the arithmetic is waiting on memory. */
    def intensity: Double = if deviceBytes == 0 then Double.NaN else ops.toDouble / deviceBytes

  private val Float4 = 4

  /** Arithmetic nodes in the DAG, after common-subexpression elimination — what the generated kernel actually executes, which is why it is counted
    * the same way `Expr.nodeCount` counts nodes rather than by walking the tree naively.
    *
    * A transcendental counts as one. It is not one: an OpenCL `sin` is tens of operations in hardware and a `StrictMath.sin` is far more than that on
    * the JVM. Weighting them would need a per-device table nobody can verify, so the honest choice is to count operations, call the result operations
    * rather than FLOPs, and read the compute-bound rows as a lower bound.
    */
  def opCount(e: Expr): Int =
    def go(e: Expr, seen: Set[Expr]): Set[Expr] =
      if seen.contains(e) then seen
      else
        e match
          case Expr.Bin(_, l, r) => go(r, go(l, seen + e))
          case Expr.Un(_, a)     => go(a, seen + e)
          case Expr.Sum(a)       => go(a, seen + e)
          case _                 => seen
    go(e, Set.empty).size

  private def chainOps(depth: Int): Int = opCount(Formulas.generatorChain(depth).body)

  private lazy val heavyOps = opCount(Formulas.generatorHeavy.body)
  private lazy val elementwiseOps = opCount(Formulas.elementwise.body)
  private lazy val elemHeavyOps = opCount(Formulas.elementwiseHeavy.body)

  /** The traffic shape of one benchmark, or `None` when there is nothing meaningful to say about it.
    *
    * Keyed on the simple `Class.method` name and the JMH parameters, because that is all a report carries.
    */
  def of(name: String, params: Map[String, String]): Option[Profile] =
    val depth = params.get("depth").flatMap(_.toIntOption).getOrElse(Bench.Depth)
    name match
      // Encoding A: nothing goes up, one float per element comes back.
      case "GeneratorBench.openclChain"                              => Some(Profile(0, Float4, 0, Float4, chainOps(depth)))
      case "GeneratorBench.openclHeavy"                              => Some(Profile(0, Float4, 0, Float4, heavyOps))
      case "GeneratorBench.spireChain" | "GeneratorBench.plainChain" => Some(Profile(0, Float4, 0, 0, chainOps(depth)))
      case "GeneratorBench.spireHeavy" | "GeneratorBench.plainHeavy" => Some(Profile(0, Float4, 0, 0, heavyOps))
      /* Breeze's `tabulate` has no in-place form, so the vector it returns is freshly allocated every call.
       * Counted as a write because that is what it costs, and flagged in the report rather than hidden. */
      case "GeneratorBench.breezeChain" => Some(Profile(0, Float4, 0, 0, chainOps(depth)))

      // Encoding B: three floats up per element, every launch, plus the result back.
      case "ElementwiseBench.openclPacked" => Some(Profile(3 * Float4, Float4, 3 * Float4, Float4, elementwiseOps))
      case "ElementwiseBench.openclHeavy"  => Some(Profile(3 * Float4, Float4, 3 * Float4, Float4, elemHeavyOps))
      // The interleave is a full host-side pass over both sides before any of it moves.
      case "ElementwiseBench.openclPacking" =>
        Some(Profile(3 * Float4, Float4, 3 * Float4, Float4, elementwiseOps))

      // Encoding C: the arrays are already there, so nothing goes up; mapping removes the copy back as well.
      case "ElementwiseBench.openclInput"       => Some(Profile(3 * Float4, Float4, 0, Float4, elementwiseOps))
      case "ElementwiseBench.openclInputHeavy"  => Some(Profile(3 * Float4, Float4, 0, Float4, elemHeavyOps))
      case "ElementwiseBench.openclInputMapped" => Some(Profile(3 * Float4, Float4, 0, 0, elementwiseOps))

      case "ElementwiseBench.spireElementwise" | "ElementwiseBench.plainElementwise" =>
        Some(Profile(3 * Float4, Float4, 0, 0, elementwiseOps))
      case "ElementwiseBench.spireHeavy" | "ElementwiseBench.plainHeavy" =>
        Some(Profile(3 * Float4, Float4, 0, 0, elemHeavyOps))
      /* `(a *:* b) + c` is two vectorised passes with a temporary between them, not one fused loop:
       * read a,b write t, then read t,c write out. Modelling it as one pass would flatter it. */
      case "ElementwiseBench.breezeElementwise" => Some(Profile(4 * Float4, 2 * Float4, 0, 0, elementwiseOps))
      case "ElementwiseBench.breezeHeavy"       => Some(Profile(3 * Float4, Float4, 0, 0, elemHeavyOps))

      case "DepthSweepBench.opencl"                          => Some(Profile(0, Float4, 0, Float4, chainOps(depth)))
      case "DepthSweepBench.spire" | "DepthSweepBench.plain" => Some(Profile(0, Float4, 0, 0, chainOps(depth)))

      case _ => None

  /** How many array elements one invocation of a benchmark covers. The depth sweep holds its size fixed rather than taking it from a parameter. */
  def elements(name: String, params: Map[String, String]): Option[Long] =
    if name.startsWith("DepthSweepBench.") then Some(DepthSweep.Size.toLong)
    else params.get("size").flatMap(_.toLongOption)
