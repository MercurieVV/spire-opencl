package io.github.mercurievv.spireopencl.algebra

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

  given OrderS[Double]:

    extension (a: Double)
      def greaterThan(a2: Double): Double = if a > a2 then 1.0 else 0.0
      infix def <<(a2: Double): Double = if a < a2 then 1.0 else 0.0
