package io.github.mercurievv.spireopencl.symbolic

import _root_.algebra.ring.Field
import io.github.mercurievv.spireopencl.algebra.{OrderS, TrigonometryCC}
import spire.math.{Algebraic, ConvertableFrom, ConvertableTo, Rational, Real}

/** The whole symbolic backend: ordinary spire/algebra instances, at `V = Expr`.
  *
  * Nothing here knows about kernels or devices. Summoning these instead of a numeric backend's is what makes the *same* polymorphic code build a tree
  * — that is the entire trick, and it is why the code being reified needs no knowledge of this library.
  *
  * `spire.algebra.Trig[Expr]` is deliberately absent: it requires `exp`, `log`, `atan` and the rest, and the IR has only `sin`. A stub that threw
  * would let code compile and fail at reification instead of at the call.
  */
object instances:

  given fieldExpr: Field[Expr]:
    override def zero: Expr = Expr.zero
    override def one: Expr = Expr.one
    override def plus(x: Expr, y: Expr): Expr = Expr.add(x, y)
    override def times(x: Expr, y: Expr): Expr = Expr.mul(x, y)
    override def div(x: Expr, y: Expr): Expr = Expr.div(x, y)
    override def negate(x: Expr): Expr = Expr.neg(x)

  given trigonometryCCExpr: TrigonometryCC[Expr] = TrigonometryCC[Expr](Expr.sin)

  given orderExpr: OrderS[Expr]:
    extension (a: Expr)
      def greaterThan(a2: Expr): Expr = Expr.gt(a, a2)
      infix def <<(a2: Expr): Expr = Expr.lt(a, a2)

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
