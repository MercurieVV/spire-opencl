package io.github.mercurievv.spireopencl.symbolic

/** Binary operations the IR can express. `Gt`/`Lt` are the comparison masks `OrderS` produces: they yield 1 or 0 as a number, not a boolean, because
  * that is how a gate or an envelope uses them.
  */
enum BinOp derives CanEqual:
  case Add, Mul, Div, Gt, Lt

enum UnOp derives CanEqual:
  case Neg, Sin

/** The intermediate representation: what a computation *is*, before anything decides how to run it.
  *
  * Numeric code that is polymorphic over its value type builds this tree when instantiated at `Expr`, from the very same source that computes numbers
  * at `Double`. Nothing in here knows what the numbers mean.
  *
  * The three ways a value enters a kernel, and they differ in how often they vary:
  *
  *   - `Const` — fixed when the formula is built; folded and inlined.
  *   - `Uniform` — one scalar per launch, identical for every work-item: a kernel argument.
  *   - `Param` — one scalar per **batch element** (dimension 1), read from the packed parameter buffer.
  *   - `Index` — the dimension-0 work-item index, as a float. Everything that varies *within* a launch is derived from this, in the IR, by the
  *     caller — the library has no notion of what dimension 0 counts.
  */
enum Expr derives CanEqual:
  case Const(v: Double)
  case Uniform(name: String)
  case Param(name: String)
  case Index
  case Bin(op: BinOp, l: Expr, r: Expr)
  case Un(op: UnOp, a: Expr)

  /** Sum of `body` over the batch dimension — a reduction, as an IR node.
    *
    * Reduction is `Field.plus` folded over dimension 1, the same algebra as everything else in the tree, so it belongs in the tree: expressing it here
    * is what lets a backend fold the reduction into the same pass that computes the elements. The batch size is a run-time value, not part of the IR,
    * so this node says "sum over whatever elements are present" and the backend supplies how many.
    *
    * Valid only at the root of a formula. Nesting has no meaning — there is one batch dimension.
    */
  case Sum(body: Expr)

object Expr:
  import BinOp.*
  import UnOp.*

  val zero: Expr = Const(0.0)
  val one: Expr = Const(1.0)

  /** Smart constructors. Folding constants and dropping identities here costs nothing and matters: a 400-term stack builds `x * 2`, `y / 2` and
    * similar for every term, and every folded node is one less line of kernel source.
    */
  def add(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b))      => Const(a + b)
    case (Const(a), _) if a == 0.0 => r
    case (_, Const(b)) if b == 0.0 => l
    case _                         => Bin(Add, l, r)

  def mul(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b))      => Const(a * b)
    case (Const(a), _) if a == 0.0 => zero
    case (_, Const(b)) if b == 0.0 => zero
    case (Const(a), _) if a == 1.0 => r
    case (_, Const(b)) if b == 1.0 => l
    case _                         => Bin(Mul, l, r)

  def div(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b)) if b != 0.0 => Const(a / b)
    case (Const(a), _) if a == 0.0        => zero
    case (_, Const(b)) if b == 1.0        => l
    case _                                => Bin(Div, l, r)

  def neg(a: Expr): Expr = a match
    case Const(v)       => Const(-v)
    case Un(Neg, inner) => inner
    case _              => Un(Neg, a)

  def sin(a: Expr): Expr = a match
    case Const(v) => Const(math.sin(v))
    case _        => Un(Sin, a)

  def gt(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b)) => if a > b then one else zero
    case _                    => Bin(Gt, l, r)

  def lt(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b)) => if a < b then one else zero
    case _                    => Bin(Lt, l, r)

  /** Reference interpreter. Used by tests to check the emitted kernel against the tree it came from, and as the fallback when no OpenCL device is
    * available. `env` resolves uniforms and params by name; `index` is the dimension-0 position.
    */
  def eval(env: String => Double, index: Double)(e: Expr): Double = e match
    case Const(v)      => v
    case Uniform(name) => env(name)
    case Param(name)   => env(name)
    case Index         => index
    case Sum(_)        => throw new IllegalArgumentException("Sum spans the batch dimension; use evalSummed")
    case Un(Neg, a)    => -eval(env, index)(a)
    case Un(Sin, a)    => math.sin(eval(env, index)(a))
    case Bin(op, l, r) =>
      val a = eval(env, index)(l)
      val b = eval(env, index)(r)
      op match
        case Add => a + b
        case Mul => a * b
        case Div => a / b
        case Gt  => if a > b then 1.0 else 0.0
        case Lt  => if a < b then 1.0 else 0.0

  /** Reference evaluation of a reduced formula: one env per batch element, each resolving that element's params and the shared uniforms. */
  def evalSummed(envs: Seq[String => Double], index: Double)(e: Expr): Double = e match
    case Sum(body) => envs.foldLeft(0.0)((acc, env) => acc + eval(env, index)(body))
    case Bin(op, l, r) if containsSum(e) =>
      val a = evalSummed(envs, index)(l)
      val b = evalSummed(envs, index)(r)
      op match
        case Add => a + b
        case Mul => a * b
        case Div => a / b
        case Gt  => if a > b then 1.0 else 0.0
        case Lt  => if a < b then 1.0 else 0.0
    case Un(op, a) if containsSum(e) =>
      val x = evalSummed(envs, index)(a)
      op match
        case Neg => -x
        case Sin => math.sin(x)
    case other => envs.headOption.fold(0.0)(env => eval(env, index)(other))

  private def containsSum(e: Expr): Boolean = e match
    case Sum(_)       => true
    case Bin(_, l, r) => containsSum(l) || containsSum(r)
    case Un(_, a)     => containsSum(a)
    case _            => false

  /** Distinct nodes in the DAG — what the kernel actually costs after common-subexpression elimination. */
  def nodeCount(e: Expr): Int =
    def go(e: Expr, seen: Set[Expr]): Set[Expr] =
      if seen.contains(e) then seen
      else
        val withSelf = seen + e
        e match
          case Bin(_, l, r) => go(r, go(l, withSelf))
          case Un(_, a)     => go(a, withSelf)
          case Sum(a)       => go(a, withSelf)
          case _            => withSelf
    go(e, Set.empty).size

  /** Every free name the tree reads, in first-encountered order. */
  def names(e: Expr): List[String] =
    def go(e: Expr, acc: List[String]): List[String] = e match
      case Uniform(n)   => if acc.contains(n) then acc else acc :+ n
      case Param(n)     => if acc.contains(n) then acc else acc :+ n
      case Bin(_, l, r) => go(r, go(l, acc))
      case Un(_, a)     => go(a, acc)
      case Sum(a)       => go(a, acc)
      case Index        => acc
      case Const(_)     => acc
    go(e, Nil)
