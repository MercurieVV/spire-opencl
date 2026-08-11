package io.github.mercurievv.spireopencl.symbolic

/** A reified computation: the tree, plus the arguments it expects at run time.
  *
  * `uniforms` and `params` are *declared*, not discovered. Constant folding can delete a name from the tree (multiply by a zero mask and it
  * vanishes), and if the argument list were derived from the tree the kernel's argument indices would shift under it. Declared order is the binding
  * order, always.
  *
  *   - `uniforms` — one value per launch, shared by every work-item; passed as scalar kernel arguments.
  *   - `params` — one value per batch element; packed into a single buffer, element-major.
  *   - `states` — one value per batch element that survives to the next launch, with `updates` giving each cell's new value.
  */
final case class Formula(
  body: Expr,
  uniforms: List[String],
  params: List[String],
  states: List[String] = Nil,
  updates: Map[String, Expr] = Map.empty)
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

  private def lookup(kind: String, declared: List[String], node: String => Expr): String => Expr =
    val table = declared.map(n => n -> node(n)).toMap
    name =>
      table.getOrElse(
        name,
        throw new IllegalArgumentException(s"undeclared $kind '$name'; declared: $declared"),
      )
