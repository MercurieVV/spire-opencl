package io.github.mercurievv.spireopencl.algebra

import cats.Functor
import cats.syntax.functor.toFunctorOps

/** Sine, as a value rather than a method.
  *
  * spire's own `Trig` is not usable here: it demands the whole transcendental suite — `exp`, `log`, `atan`, `sinh` — and a symbolic backend can only
  * honour what its IR can express. Carrying the one function as data keeps the requirement honest and lets any type join by supplying it.
  */
case class TrigonometryCC[A](sin: A => A)

object TrigonometryCC:
  given TrigonometryCC[Double] = TrigonometryCC[Double](Math.sin)

  /** Pointwise over any functor: a container of values is trigonometric if its elements are. */
  given functorTrigonometryCC: [V: TrigonometryCC, S[_]: Functor] => TrigonometryCC[S[V]] =
    TrigonometryCC[S[V]](_.map(summon[TrigonometryCC[V]].sin))

/** The syntax side of [[TrigonometryCC]]: `a.sin` for any `A` that has one. */
trait Trigonometry[A: TrigonometryCC]:
  extension (a: A) def sin: A

object Trigonometry:
  /** In the companion so it resolves without an import, wherever a `TrigonometryCC` is in scope. */
  given [A: TrigonometryCC] => Trigonometry[A]:
    extension (a: A) def sin: A = summon[TrigonometryCC[A]].sin.apply(a)
