package io.github.mercurievv.spireopencl.symbolic

import cats.Id
import cats.data.StateT
import io.github.mercurievv.spireopencl.symbolic.state.{Store, VarId}

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
  *
  * `inputs` comes last so that a formula that gains one keeps every earlier argument's meaning, and every existing caller keeps compiling.
  */
final case class Formula(
  body: Expr,
  uniforms: List[String],
  params: List[String],
  states: List[String] = Nil,
  updates: Map[String, Expr] = Map.empty,
  inputs: List[String] = Nil)
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

  private def lookup(kind: String, declared: List[String], node: String => Expr): String => Expr =
    val table = declared.map(n => n -> node(n)).toMap
    name =>
      table.getOrElse(
        name,
        throw new IllegalArgumentException(s"undeclared $kind '$name'; declared: $declared"),
      )
