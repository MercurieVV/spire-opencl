package io.github.mercurievv.spireopencl.symbolic.state

import algebra.ring.Field
import cats.Monad
import cats.data.StateT
import cats.syntax.functor.*

/** State that survives from one launch to the next.
  *
  * Three things are deliberately separate here, because bundling them is what made earlier attempts at this awkward:
  *
  *   - **the cell** — `Var`, one value with a pure step. A cell knows only its own `V`: no identity, no map, no backend, so the same step function
  *     serves an oscillator's phase, a filter's memory or a latch.
  *   - **the address** — `At`, which places a cell in a `Store`. Two cells differ by their id and by nothing else.
  *   - **the program `F`** — supplied by the caller (e.g. a reader over per-launch globals). This module knows nothing about what `F` is; it only
  *     needs `Monad[F]` to thread one cell's step into a larger state.
  *
  * Nothing here is a `cats.effect.Ref`. `Ref.modify` takes a function to run at run time; a step function here is consumed while the signal graph is
  * being *composed* — at `V = Expr` it becomes an `Expr.State` read plus an entry in `Formula.updates`, and never runs on the host at all.
  */
type VarId = Int

/** Every named cell a program carries, keyed by id. */
type Store[V] = Map[VarId, V]

/** One cell and its step: previous value in, next value and an output out, with `F` supplying whatever context the step needs to compute them. */
type Var[F[_], V, A] = StateT[F, V, A]

/** Where a cell lives inside a larger state. Deliberately not tied to `Map`: what matters is that an id selects one `V` and puts it back. */
trait At[S, K, V]:
  def get(s: S, k: K): V
  def set(s: S, k: K, v: V): S

object At:

  /** A cell never written reads as zero — which is also what a device's state buffer holds before its first launch, so a host and a device backend
    * agree on the initial value without either of them having to declare it.
    */
  given [K, V] => (F: Field[V]) => At[Map[K, V], K, V]:
    def get(s: Map[K, V], k: K): V = s.getOrElse(k, F.zero)
    def set(s: Map[K, V], k: K, v: V): Map[K, V] = s.updated(k, v)

extension [F[_], V, A](cell: Var[F, V, A])

  /** Place this cell at `k`. The zoom between "a step that knows one value" and "a program that threads them all". */
  def at[S](k: VarId)(using at: At[S, VarId, V], F: Monad[F]): StateT[F, S, A] =
    StateT[F, S, A](s => cell.run(at.get(s, k)).map((v, a) => (at.set(s, k, v), a)))
