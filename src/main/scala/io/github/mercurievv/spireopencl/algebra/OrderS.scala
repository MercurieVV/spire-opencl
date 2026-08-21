package io.github.mercurievv.spireopencl.algebra

import _root_.algebra.ring.Field
import spire.algebra.Order

/** Comparison that returns a **value of the same type**, not a `Boolean`.
  *
  * `a >> b` is 1 where the comparison holds and 0 where it does not, so a comparison can be multiplied into an expression as a mask. That is what
  * makes envelopes and gates ordinary arithmetic, and it is also the only form a data-parallel backend can use: a branch per element is not a
  * comparison a vectorised or GPU kernel wants to make.
  */
trait OrderS[@specialized(Int, Long, Float, Double) A]:

  extension (a: A)
    def greaterThan(a2: A): A
    infix def >>(a2: A): A = greaterThan(a2)
    infix def <<(a2: A): A

object OrderS:

  /** Any ordered field gets masking for free — `Double`, `Float`, `Rational`, whatever `V` a caller picks — instead of pinning this to `Double`. */
  given orderedField[V](using F: Field[V], O: Order[V]): OrderS[V] with

    extension (a: V)
      def greaterThan(a2: V): V = if O.gt(a, a2) then F.one else F.zero
      infix def <<(a2: V): V = if O.lt(a, a2) then F.one else F.zero
