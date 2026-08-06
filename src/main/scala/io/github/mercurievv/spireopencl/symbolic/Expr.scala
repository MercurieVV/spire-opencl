package io.github.mercurievv.spireopencl.symbolic

/** Binary operations the IR can express, each carrying the arithmetic it means.
  *
  * The function is the single definition of the operation: constant folding, the reference interpreter and the differential tests all read it, so
  * there is no second place for the meaning of `Div` to drift. Only the *rendering* lives in `CodeGen`.
  *
  * `Gt`/`Lt` are comparison **masks**: they yield 1 or 0 as a number, not a boolean, because that is what can be multiplied into an expression and
  * what a data-parallel backend wants instead of a branch per element.
  */
enum BinOp(val eval: (Double, Double) => Double) derives CanEqual:
  case Add extends BinOp(_ + _)
  case Mul extends BinOp(_ * _)
  case Div extends BinOp(_ / _)
  case Gt extends BinOp((a, b) => if a > b then 1.0 else 0.0)
  case Lt extends BinOp((a, b) => if a < b then 1.0 else 0.0)
  case Pow extends BinOp(math.pow)
  case Atan2 extends BinOp(math.atan2)

/** Unary operations, likewise carrying their arithmetic. The transcendental set is what `spire.algebra.Trig` and `NRoot` require — the point being
  * that a caller writes ordinary spire code and this is what it turns into.
  */
enum UnOp(val eval: Double => Double) derives CanEqual:
  case Neg extends UnOp(-_)
  case Sin extends UnOp(math.sin)
  case Cos extends UnOp(math.cos)
  case Tan extends UnOp(math.tan)
  case Asin extends UnOp(math.asin)
  case Acos extends UnOp(math.acos)
  case Atan extends UnOp(math.atan)
  case Sinh extends UnOp(math.sinh)
  case Cosh extends UnOp(math.cosh)
  case Tanh extends UnOp(math.tanh)
  case Exp extends UnOp(math.exp)
  case Expm1 extends UnOp(math.expm1)
  case Log extends UnOp(math.log)
  case Log1p extends UnOp(math.log1p)
  case Sqrt extends UnOp(math.sqrt)

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

  def gt(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b)) => if a > b then one else zero
    case _                    => Bin(Gt, l, r)

  def lt(l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b)) => if a < b then one else zero
    case _                    => Bin(Lt, l, r)

  /** Any unary operation, folded when its argument is already known. The operation's own `eval` does the folding, so a new op cannot arrive with a
    * constant-folding rule that disagrees with how it is interpreted.
    */
  def un(op: UnOp, a: Expr): Expr = a match
    case Const(v) => Const(op.eval(v))
    case _        => Un(op, a)

  def bin(op: BinOp, l: Expr, r: Expr): Expr = (l, r) match
    case (Const(a), Const(b)) => Const(op.eval(a, b))
    case _                    => Bin(op, l, r)

  def sin(a: Expr): Expr = un(Sin, a)
  def cos(a: Expr): Expr = un(Cos, a)
  def sqrt(a: Expr): Expr = un(Sqrt, a)

  /** `x^n` for a small whole `n` is repeated multiplication, which every backend does better than a general `pow` — and it keeps the tree in the
    * subset a code generator without `pow` could still handle.
    */
  def pow(base: Expr, exponent: Expr): Expr = (base, exponent) match
    case (_, Const(1.0))                                 => base
    case (_, Const(0.0))                                 => one
    case (_, Const(n)) if n == n.round.toDouble && math.abs(n) <= 8.0 && n > 0 =>
      (1 until n.toInt).foldLeft(base)((acc, _) => mul(acc, base))
    case _ => bin(Pow, base, exponent)

  /** Reference interpreter. Used by tests to check the emitted kernel against the tree it came from, and as the fallback when no OpenCL device is
    * available. `env` resolves uniforms and params by name; `index` is the dimension-0 position.
    */
  def eval(env: String => Double, index: Double)(e: Expr): Double = e match
    case Const(v)      => v
    case Uniform(name) => env(name)
    case Param(name)   => env(name)
    case Index         => index
    case Sum(_)        => throw new IllegalArgumentException("Sum spans the batch dimension; use evalSummed")
    case Un(op, a)     => op.eval(eval(env, index)(a))
    case Bin(op, l, r) => op.eval(eval(env, index)(l), eval(env, index)(r))

  /** Reference evaluation of a reduced formula: one env per batch element, each resolving that element's params and the shared uniforms. */
  def evalSummed(envs: Seq[String => Double], index: Double)(e: Expr): Double = e match
    case Sum(body) => envs.foldLeft(0.0)((acc, env) => acc + eval(env, index)(body))
    case Bin(op, l, r) if containsSum(e) => op.eval(evalSummed(envs, index)(l), evalSummed(envs, index)(r))
    case Un(op, a) if containsSum(e)     => op.eval(evalSummed(envs, index)(a))
    case other                           => envs.headOption.fold(0.0)(env => eval(env, index)(other))

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
