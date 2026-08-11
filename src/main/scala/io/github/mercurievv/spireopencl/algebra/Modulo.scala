package io.github.mercurievv.spireopencl.algebra

/** Remainder, which spire's `Field` does not offer and a periodic accumulator cannot do without.
  *
  * A value that is only ever added to grows without bound, and absolute precision falls as it grows — in single precision, fast enough to matter
  * within seconds at audio rates. Folding it back into one period each step keeps a stored value small forever, at the cost of one `fmod`.
  */
trait Modulo[V]:
  def mod(x: V, m: V): V

object Modulo:
  def apply[V](using m: Modulo[V]): Modulo[V] = m

  given Modulo[Double] = _ % _
