package io.github.mercurievv.spireopencl.symbolic

import _root_.algebra.ring.Field
import io.github.mercurievv.spireopencl.algebra.{Modulo, OrderS}
import spire.algebra.{NRoot, Trig}
import spire.math.{Algebraic, ConvertableFrom, ConvertableTo, Rational, Real}

/** The whole symbolic backend: ordinary spire/algebra instances, at `V = Expr`.
  *
  * Nothing here knows about kernels or devices. Summoning these instead of a numeric backend's is what makes the *same* polymorphic code build a tree
  * — that is the entire trick, and it is why the code being reified needs no knowledge of this library.
  *
  * With these in scope, `spire.math.sin(d)`, `a * b - c`, `a ** 3` and `sqrt(x)` are ordinary spire and produce a tree:
  *
  * ```scala
  * import spire.implicits.*, spire.math.sin
  * import io.github.mercurievv.spireopencl.symbolic.instances.given
  *
  * def program[V: {Field, Trig}](b: V, c: V, d: V): V = b * c - sin(d)
  * ```
  *
  * `program[Double]` computes a number; `program[Expr]` gives the expression that number came from, ready to compile.
  */
object instances:

  given fieldExpr: Field[Expr]:
    override def zero: Expr = Expr.zero
    override def one: Expr = Expr.one
    override def plus(x: Expr, y: Expr): Expr = Expr.add(x, y)
    override def times(x: Expr, y: Expr): Expr = Expr.mul(x, y)
    override def div(x: Expr, y: Expr): Expr = Expr.div(x, y)
    override def negate(x: Expr): Expr = Expr.neg(x)

  /** The reason a caller can write plain spire and get a kernel. Every method here is an OpenCL built-in with the same semantics, so nothing is
    * approximated on the way down; `toRadians`/`toDegrees` are a multiply and fold into a constant.
    */
  given trigExpr: Trig[Expr]:
    def e: Expr = Expr.Const(math.E)
    def pi: Expr = Expr.Const(math.Pi)
    def exp(a: Expr): Expr = Expr.un(UnOp.Exp, a)
    def expm1(a: Expr): Expr = Expr.un(UnOp.Expm1, a)
    def log(a: Expr): Expr = Expr.un(UnOp.Log, a)
    def log1p(a: Expr): Expr = Expr.un(UnOp.Log1p, a)
    def sin(a: Expr): Expr = Expr.un(UnOp.Sin, a)
    def cos(a: Expr): Expr = Expr.un(UnOp.Cos, a)
    def tan(a: Expr): Expr = Expr.un(UnOp.Tan, a)
    def asin(a: Expr): Expr = Expr.un(UnOp.Asin, a)
    def acos(a: Expr): Expr = Expr.un(UnOp.Acos, a)
    def atan(a: Expr): Expr = Expr.un(UnOp.Atan, a)
    def atan2(y: Expr, x: Expr): Expr = Expr.bin(BinOp.Atan2, y, x)
    def sinh(x: Expr): Expr = Expr.un(UnOp.Sinh, x)
    def cosh(x: Expr): Expr = Expr.un(UnOp.Cosh, x)
    def tanh(x: Expr): Expr = Expr.un(UnOp.Tanh, x)
    def toRadians(a: Expr): Expr = Expr.mul(a, Expr.Const(math.Pi / 180.0))
    def toDegrees(a: Expr): Expr = Expr.mul(a, Expr.Const(180.0 / math.Pi))

  /** `sqrt`, `nroot` and `fpow`. An integer power folds to repeated multiplication (see `Expr.pow`) rather than a `pow` call. */
  given nrootExpr: NRoot[Expr]:

    def nroot(a: Expr, n: Int): Expr =
      if n == 2 then Expr.un(UnOp.Sqrt, a) else Expr.pow(a, Expr.Const(1.0 / n.toDouble))
    override def sqrt(a: Expr): Expr = Expr.un(UnOp.Sqrt, a)
    def fpow(a: Expr, b: Expr): Expr = Expr.pow(a, b)

  given orderExpr: OrderS[Expr]:

    extension (a: Expr)
      def greaterThan(a2: Expr): Expr = Expr.gt(a, a2)
      infix def <<(a2: Expr): Expr = Expr.lt(a, a2)

  given moduloExpr: Modulo[Expr] = Expr.rem(_, _)

  /** Numeric literals enter the tree here. The exact-arithmetic conversions have no meaning for a float kernel, but they cost nothing to support at
    * `Double` precision, and refusing them would reject ordinary code for no gain.
    */
  given convertableToExpr: ConvertableTo[Expr]:
    override def fromByte(n: Byte): Expr = Expr.Const(n.toDouble)
    override def fromShort(n: Short): Expr = Expr.Const(n.toDouble)
    override def fromInt(n: Int): Expr = Expr.Const(n.toDouble)
    override def fromLong(n: Long): Expr = Expr.Const(n.toDouble)
    override def fromFloat(n: Float): Expr = Expr.Const(n.toDouble)
    override def fromDouble(n: Double): Expr = Expr.Const(n)
    override def fromBigInt(n: BigInt): Expr = Expr.Const(n.toDouble)
    override def fromBigDecimal(n: BigDecimal): Expr = Expr.Const(n.toDouble)
    override def fromRational(n: Rational): Expr = Expr.Const(n.toDouble)
    override def fromAlgebraic(n: Algebraic): Expr = Expr.Const(n.toDouble)
    override def fromReal(n: Real): Expr = Expr.Const(n.toDouble)
    override def fromType[B: ConvertableFrom](b: B): Expr = Expr.Const(ConvertableFrom[B].toDouble(b))
