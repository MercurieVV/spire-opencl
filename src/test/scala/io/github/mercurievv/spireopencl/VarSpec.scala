package io.github.mercurievv.spireopencl

import cats.Id
import cats.data.StateT
import io.github.mercurievv.spireopencl.symbolic.{Expr, instances}
import io.github.mercurievv.spireopencl.symbolic.state.{At, Store, Var, VarId, at}
import spire.implicits.DoubleAlgebra
import weaver.*

/** `Var`/`Store`/`At` on their own: a named cell's step, zoomed into a larger state, independent of any backend. */
object VarSpec extends SimpleIOSuite:

  import instances.given

  pureTest("a cell never written reads as the field's zero") {
    val store: Store[Double] = Map.empty
    expect(summon[At[Store[Double], VarId, Double]].get(store, 7) == 0.0)
  }

  pureTest(".at zooms a lone step into a larger store, and leaves other keys untouched") {
    val increment: Var[Id, Double, Double] = StateT(prev => (prev + 1.0, prev))
    val zoomed = increment.at[Store[Double]](5)

    val (afterOne, firstOut) = zoomed.run(Map(9 -> 42.0))
    val (afterTwo, secondOut) = zoomed.run(afterOne)

    expect(firstOut == 0.0) &&
    expect(secondOut == 1.0) &&
    expect(afterTwo == Map(9 -> 42.0, 5 -> 2.0))
  }

  pureTest("two cells at different ids are independent") {
    val incrementBy = (n: Double) => StateT[Id, Double, Unit](prev => (prev + n, ()))
    val program =
      for
        _ <- incrementBy(1.0).at[Store[Double]](0)
        _ <- incrementBy(10.0).at[Store[Double]](1)
        _ <- incrementBy(1.0).at[Store[Double]](0)
      yield ()

    val (finalStore, _) = program.run(Map.empty)
    expect(finalStore == Map(0 -> 2.0, 1 -> 10.0))
  }

  pureTest("the same mechanism at V = Expr builds the tree Formula.updates expects") {
    // Exactly the shape AudioFormula.voice relies on: a step that reads the previous cell value and an Expr.Uniform,
    // and returns the new value as an expression rather than a number — the update Reify.stateful wants.
    val advancePhase: Var[Id, Expr, Expr] =
      StateT(prev => (Expr.add(prev, Expr.Uniform("dt")), prev))

    val (nextStore, phaseReadThisLaunch) = advancePhase.at[Store[Expr]](0).run(Map.empty)

    expect(phaseReadThisLaunch == Expr.Const(0.0)) &&
    expect(nextStore(0) == Expr.add(Expr.Const(0.0), Expr.Uniform("dt")))
  }
