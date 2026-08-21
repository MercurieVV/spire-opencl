package io.github.mercurievv.spireopencl.symbolic

import cats.Id
import cats.data.StateT
import io.github.mercurievv.spireopencl.symbolic.state.{Store, VarId}
import io.github.mercurievv.spireopencl.util.FieldLabels

import scala.deriving.Mirror

/** A reified computation: the tree, plus the arguments it expects at run time.
  *
  * `uniforms` and `params` are *declared*, not discovered. Constant folding can delete a name from the tree (multiply by a zero mask and it
  * vanishes), and if the argument list were derived from the tree the kernel's argument indices would shift under it. Declared order is the binding
  * order, always.
  *
  *   - `uniforms` — one value per launch, shared by every work-item; passed as scalar kernel arguments.
  *   - `params` — one value per batch element; packed into a single buffer, element-major.
  *   - `states` — one value per batch element that survives to the next launch, with `updates` giving each cell's new value.
  *   - `inputs` — one value per work-item: a device-resident array, one buffer each, written by the host independently of launching.
  *   - `extraOutputs` — additional per-element results, each its own named device buffer, computed and written alongside `body` in the same launch.
  *     `body` stays the one output every other operation (`summed`, `scaledBy`, a caller reading `render`'s plain `out`) already knows about; a
  *     formula that needs more than one result declares the rest here rather than every reader learning to expect a list.
  *
  * `inputs` comes last so that a formula that gains one keeps every earlier argument's meaning, and every existing caller keeps compiling.
  */
final case class Formula(
  body: Expr,
  uniforms: List[String],
  params: List[String],
  states: List[String] = Nil,
  updates: Map[String, Expr] = Map.empty,
  inputs: List[String] = Nil,
  extraOutputs: List[(String, Expr)] = Nil)
    derives CanEqual:

  require(
    updates.keySet == states.toSet,
    s"every state needs exactly one update; declared ${states.sorted}, updated ${updates.keys.toList.sorted}",
  )

  updates.foreach: (name, update) =>
    require(
      !Expr.containsIndex(update),
      s"state update '$name' depends on Index; there is no single work-item at which to take it",
    )
    require(
      !Expr.containsSum(update),
      s"state update '$name' contains a Sum; state is per batch element, reduction spans them",
    )

  require(
    extraOutputs.isEmpty || sum.isEmpty,
    "extraOutputs are not supported on a reduced formula: reduction collapses the batch dimension for body alone, " +
      "and there is no matching total for the rest",
  )

  def nodeCount: Int = Expr.nodeCount(body)

  /** The reduced part of this formula, if it has one. Everything above it in the tree operates on the finished sum. */
  def sum: Option[Expr] =
    def find(e: Expr): Option[Expr] = e match
      case Expr.Sum(inner)   => Some(inner)
      case Expr.Bin(_, l, r) => find(l).orElse(find(r))
      case Expr.Un(_, a)     => find(a)
      case _                 => None
    find(body)

  /** Whether this formula produces the sum over the batch rather than one element's values. */
  def isReduced: Boolean = sum.isDefined

  /** Sum this formula over the batch dimension. The parameters are unchanged — they are still per-element, now read once per element per work-item
    * instead of once per work-item.
    */
  def summed: Formula = if isReduced then this else copy(body = Expr.Sum(body))

  /** Scale the result. Applied above the `Expr.Sum` when there is one, so it costs one multiply per work-item rather than one per batch element. */
  def scaledBy(gain: Double): Formula = copy(body = Expr.mul(body, Expr.Const(gain)))

/** A `Formula` tagged, at the type level only, with the case class its uniforms and params were declared from — `F` here is a phantom, carried but
  * never inspected.
  *
  * Why this exists: `formula.uniforms`/`formula.params` are plain `List[String]`, so nothing stops compiling a `Kernel` from one formula and then
  * launching it with a case class meant for a different one — a mismatch that only shows up as a runtime "missing field" once the names happen not to
  * line up, or worse, not at all when they coincidentally do. `ClKernel.compileT` turns this `F` into the one type its `TypedKernel.renderUnsafeT`
  * will accept, so a formula built for `Args` and a launch built for some other case class fail to compile rather than fail — or silently succeed on
  * the wrong data — at run time.
  */
final case class TypedFormula[F[_]](formula: Formula)

/** A `Formula` producing more than one named result in a single launch — see `Reify.outTyped`. `In` tags the arguments the same way
  * `TypedFormula[In]` does; `Out` tags which `Expr` `build` returned for which result, by `Out`'s own field labels, in `outputNames`.
  *
  * `body` holds `outputNames.head`'s expression and `extraOutputs` the rest — see `Formula`'s doc — but a caller working through
  * `TypedKernel2.renderT` never sees that split: it hands back one `Out[Float]`, reassembled by name in `outputNames`' order.
  */
final case class TypedFormula2[In[_], Out[_]](formula: Formula, outputNames: List[String])

/** Compose → IR.
  *
  * A computation is an opaque Scala function; the way to see inside it is to apply it to symbolic inputs and keep what comes back. There is no macro
  * and no inspection — reification is by application.
  */
object Reify:

  /** Name the arguments, receive them as `Expr` nodes, hand back the expression built with them. The dimension-0 index is available directly as
    * `Expr.Index`; anything that varies within a launch is derived from it here, by the caller, because the library does not know what dimension 0
    * counts.
    */
  def apply(uniforms: List[String], params: List[String])(build: (String => Expr, String => Expr) => Expr): Formula =
    Formula(
      build(
        lookup("uniform", uniforms, Expr.Uniform.apply),
        lookup("param", params, Expr.Param.apply),
      ),
      uniforms,
      params,
    )

  /** As `apply`, but `params` is read off a case class's own field labels instead of given as a list — `Reify[Point](uniforms = Nil) { (_, point) =>
    * richProgram(point) }` declares `params = List("x", "y", "z")` and builds `Point[Expr]` for `case class Point[V](x: V, y: V, z: V)`, both from
    * `Point` alone, instead of naming `"x", "y", "z"` once for `Formula` and again through `paramsAs`.
    *
    * The result is a `TypedFormula[F]`, not a bare `Formula`, so `ClKernel.compileT[Eff, F]` and `TypedKernel.renderUnsafeT` are pinned to this same
    * `F` — a launch built for some other case class fails to compile rather than reading whichever of its fields happens to share a name.
    */
  inline def apply[F[_]](
    uniforms: List[String],
  )(
    build: (String => Expr, F[Expr]) => Expr,
  )(using Mirror.ProductOf[F[String]],
    Mirror.ProductOf[F[Expr]],
  ): TypedFormula[F] =
    val params = FieldLabels.namesOf[F]
    TypedFormula(
      Formula(
        build(
          lookup("uniform", uniforms, Expr.Uniform.apply),
          FieldLabels.fill[F, Expr](lookup("param", params, Expr.Param.apply)),
        ),
        uniforms,
        params,
      ),
    )

  /** As `apply[F]`, but `build` returns `Out[Expr]` instead of a single `Expr` — one result per field of `Out`, each its own device buffer, produced
    * by the same launch. For `case class Result[V](sum: V, product: V)`, `Reify.outTyped[Point, Result](uniforms = Nil) { (_, point) =>
    * Result(point.x + point.y + point.z, point.x * point.y * point.z) }` declares two outputs, `sum` and `product`, and
    * `TypedKernel2.renderT(Point[Float](...))` hands both back as one `Result[Float]`.
    *
    * `Out`'s first field becomes `Formula.body`; the rest become `Formula.extraOutputs`, in the order `Out` declares them. Neither the split nor the
    * order is a caller's concern — `TypedFormula2.outputNames` and `TypedKernel2.renderT` reassemble `Out[Float]` by name, not by position.
    */
  inline def outTyped[In[_], Out[_]](
    uniforms: List[String],
  )(
    build: (String => Expr, In[Expr]) => Out[Expr],
  )(using Mirror.ProductOf[In[String]],
    Mirror.ProductOf[In[Expr]],
    Mirror.ProductOf[Out[String]],
  ): TypedFormula2[In, Out] =
    val params = FieldLabels.namesOf[In]
    val outNames = FieldLabels.namesOf[Out]
    val outExpr = build(
      lookup("uniform", uniforms, Expr.Uniform.apply),
      FieldLabels.fill[In, Expr](lookup("param", params, Expr.Param.apply)),
    )
    val outValues = FieldLabels.read[Out, Expr](outExpr)
    val (_, primaryExpr) = outValues.head
    TypedFormula2(
      Formula(primaryExpr, uniforms, params, extraOutputs = outValues.tail),
      outNames,
    )

  /** As `apply`, with device-resident arrays: a third lookup whose names read one element per work-item.
    *
    * This is the shape ordinary array code wants — `a * b + c` over three arrays is three `Input`s, and the arrays are uploaded once and reused
    * across launches rather than sent with each. Dimension 0 is the array, so a kernel compiled with `size = n` covers the whole thing in one launch,
    * and the batch dimension is left free for whatever it meant before.
    */
  def arrays(
    uniforms: List[String],
    params: List[String],
    inputs: List[String],
  )(
    build: (String => Expr, String => Expr, String => Expr) => Expr,
  ): Formula =
    Formula(
      build(
        lookup("uniform", uniforms, Expr.Uniform.apply),
        lookup("param", params, Expr.Param.apply),
        lookup("input", inputs, Expr.Input.apply),
      ),
      uniforms,
      params,
      inputs = inputs,
    )

  /** As `apply`, with cells that persist between launches.
    *
    * `build` receives a third lookup for reading them and returns, alongside the body, the new value of every declared cell. Updates are returned
    * rather than written because there is nothing to write to: the tree is a value, and a cell's new value is simply another expression over the same
    * inputs — including its own previous value.
    */
  def stateful(
    uniforms: List[String],
    params: List[String],
    states: List[String],
  )(
    build: (String => Expr, String => Expr, String => Expr) => (Expr, Map[String, Expr]),
  ): Formula =
    val (body, updates) = build(
      lookup("uniform", uniforms, Expr.Uniform.apply),
      lookup("param", params, Expr.Param.apply),
      lookup("state", states, Expr.State.apply),
    )
    Formula(body, uniforms, params, states, updates)

  /** As `arrays`, but `inputs` is read off a case class's own field labels instead of given as a list — the array analogue of `apply[F]`. For
    * `case class Point[V](x: V, y: V, z: V)`, `Reify.arraysTyped[Point](uniforms = Nil, params = Nil) { (_, _, point) => richProgram(point) }`
    * declares one device-resident array per field, `x`/`y`/`z`, and hands `build` the filled `Point[Expr]` instead of three named `Expr.Input`s.
    *
    * The result is a `TypedFormula[F]`, so `TypedKernel.writeInputsT` can take a `Point[Array[Float]]` — one array per field, matched by name to the
    * same fields this declared — instead of one `writeInputUnsafe` call per array.
    */
  inline def arraysTyped[F[_]](
    uniforms: List[String],
    params: List[String],
  )(
    build: (String => Expr, String => Expr, F[Expr]) => Expr,
  )(using Mirror.ProductOf[F[String]],
    Mirror.ProductOf[F[Expr]],
  ): TypedFormula[F] =
    val inputs = FieldLabels.namesOf[F]
    TypedFormula(
      Formula(
        build(
          lookup("uniform", uniforms, Expr.Uniform.apply),
          lookup("param", params, Expr.Param.apply),
          FieldLabels.fill[F, Expr](lookup("input", inputs, Expr.Input.apply)),
        ),
        uniforms,
        params,
        inputs = inputs,
      ),
    )

  /** As `stateful`, but the cells come from a `Var`/`.at`-composed program over `Store[Expr]` instead of a hand-written `(Expr, Map[String, Expr])`
    * pair — the bridge from the algebra in `symbolic.state` to a device-resident cell, so a caller does not re-derive it per timbre.
    *
    * `program` is built once, with real `Expr.Uniform`/`Expr.Param` nodes bound to the declared names, and then run twice: once from an empty store
    * to see which `VarId`s it touches — cell ids come from the program's own structure, not the folded tree, so a cell that constant-folds out of the
    * body still keeps its slot — and once seeded with each touched id's `Expr.State` read, which is what turns "the value a cell held last launch"
    * into part of the tree rather than a number that does not exist yet. A `VarId`'s device name is `s"v$$id"`.
    */
  def statefulVar(uniforms: List[String], params: List[String])(program: (String => Expr, String => Expr) => StateT[Id, Store[Expr], Expr]): Formula =
    val uniform = lookup("uniform", uniforms, Expr.Uniform.apply)
    val param = lookup("param", params, Expr.Param.apply)
    val built = program(uniform, param)
    val touched: List[VarId] = built.runS(Map.empty).keys.toList.sorted
    def name(id: VarId): String = s"v$id"
    val seeded = touched.map(id => id -> Expr.State(name(id))).toMap
    val (finalStore, body) = built.run(seeded)
    Formula(
      body,
      uniforms,
      params,
      states  = touched.map(name),
      updates = finalStore.map((id, next) => name(id) -> next),
    )

  /** As `statefulVar`, but `uniforms` and `params` both come from one case class's field labels — `uniformFields` says which of `F`'s fields are
    * uniforms, everything else is a param — and `program` reads them back off one `F[Expr]` instead of two separate lookups. The result is a
    * `TypedFormula[F]` rather than a bare `Formula`, so `ClKernel.compileT[Eff, F]` and, downstream, `TypedKernel.renderUnsafeT` are pinned to the
    * exact same `F` used here: an `Args` built for a different formula cannot be handed to this one's kernel.
    */
  inline def statefulVarTyped[F[_]](
    uniformFields: Set[String],
  )(
    program: F[Expr] => StateT[Id, Store[Expr], Expr],
  )(using Mirror.ProductOf[F[String]],
    Mirror.ProductOf[F[Expr]],
  ): TypedFormula[F] =
    val (uniformNames, paramNames) = FieldLabels.namesOf[F].partition(uniformFields)
    TypedFormula(
      statefulVar(uniformNames, paramNames) { (uniform, param) =>
        program(FieldLabels.fill[F, Expr](name => if uniformFields(name) then uniform(name) else param(name)))
      },
    )

  private def lookup(kind: String, declared: List[String], node: String => Expr): String => Expr =
    val table = declared.map(n => n -> node(n)).toMap
    name =>
      table.getOrElse(
        name,
        throw new IllegalArgumentException(s"undeclared $kind '$name'; declared: $declared"),
      )

  /** Fill a case class from a lookup instead of naming each field twice — `paramsAs[Point](p)` is `Point(p("x"), p("y"), p("z"))` for
    * `case class Point[V](x: V, y: V, z: V)`, built from the case class's own field labels in their declared order. That order is also the order
    * `params` (or `uniforms`/`inputs`) must declare on `Formula`, since it becomes the binding order there.
    *
    * Works with any of the lookups `apply`/`arrays`/`stateful` hand to `build` — `paramsAs[Point](param)`, `paramsAs[Point](uniform)`, etc.
    */
  inline def paramsAs[F[_]](lookup: String => Expr)(using Mirror.ProductOf[F[String]], Mirror.ProductOf[F[Expr]]): F[Expr] =
    FieldLabels.fill[F, Expr](lookup)
