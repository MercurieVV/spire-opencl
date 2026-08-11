package io.github.mercurievv.spireopencl

import cats.Id
import cats.data.StateT
import cats.effect.IO
import cats.syntax.apply.catsSyntaxApplyOps
import io.github.mercurievv.spireopencl.opencl.ClKernel
import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify, instances}
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

  private val size = 128

  pureTest("Reify.statefulVar discovers a cell's slot from the program's structure, not the folded body") {
    // Two cells, combined with flatMap: `kept`'s value reaches the body, `silent`'s does not — its own output is a
    // constant, thrown away by `*>`. Discovery runs the program forward once and reads the keys it touched, so
    // `silent` keeps its slot regardless; a derived-from-the-tree scheme would drop it and shift `kept`'s index
    // under it the moment a caller added or removed an unused cell.
    val kept: Var[Id, Expr, Expr] = StateT(prev => (Expr.add(prev, Expr.Uniform("step")), prev))
    val silent: Var[Id, Expr, Expr] = StateT(prev => (Expr.add(prev, Expr.Const(1.0)), Expr.zero))
    val program = (_: String => Expr, _: String => Expr) => kept.at[Store[Expr]](0) *> silent.at[Store[Expr]](1) *> kept.at[Store[Expr]](0)

    val formula = Reify.statefulVar(List("step"), Nil)(program)

    expect(formula.states.toSet == Set("v0", "v1"), s"got ${formula.states}") &&
    expect(formula.updates("v1") == Expr.add(Expr.State("v1"), Expr.Const(1.0))) &&
    expect(
      !Expr.names(formula.body).contains("v1"),
      "v1 leaked into the body despite being discarded",
    )
  }

  /** The cell's step comes from `Var`/`.at`, run through the library's own bridge from `symbolic.state` to a device-resident cell — nothing above
    * this line ever reaches a kernel, so it's the one thing worth proving on real hardware.
    */
  private val counterViaVar: Formula =
    val advance: Var[Id, Expr, Expr] = StateT(prev => (Expr.add(prev, Expr.Uniform("step")), prev))
    Reify.statefulVar(List("step"), Nil)((_, _) => advance.at[Store[Expr]](0))

  test("a Var-driven cell put/get/modifies on the actual device — not inferred from output") {
    val formula = counterViaVar

    ClKernel.compile[IO](formula, size).use { kernel =>
      IO {
        val out = new Array[Float](size)

        // get: a freshly compiled kernel's cell reads as the field's zero, exactly as At.get promises off-device.
        val initial = new Array[Float](1)
        kernel.readStateUnsafe(initial)

        // put: seed the device cell directly, bypassing every launch.
        kernel.writeStateUnsafe(Array(100.0f))
        val afterWrite = new Array[Float](1)
        kernel.readStateUnsafe(afterWrite)

        // modify: a launch reads what was put, and the update Var's step produced writes back on top of it.
        kernel.renderUnsafe(Map("step" -> 5.0f), Map.empty, out)
        val firstLaunchOut = out.clone()
        val afterOneLaunch = new Array[Float](1)
        kernel.readStateUnsafe(afterOneLaunch)

        kernel.renderUnsafe(Map("step" -> 5.0f), Map.empty, out)
        val afterTwoLaunches = new Array[Float](1)
        kernel.readStateUnsafe(afterTwoLaunches)

        expect(
          initial.toList == List(0.0f),
          s"a fresh cell should read zero, got ${initial.toList}",
        ) &&
        expect(
          afterWrite.toList == List(100.0f),
          s"the put did not stick, got ${afterWrite.toList}",
        ) &&
        expect(
          firstLaunchOut.forall(_ == 100.0f),
          "the launch's read did not see the seeded value",
        ) &&
        expect(
          afterOneLaunch.toList == List(105.0f),
          s"the step's update did not modify the cell, got ${afterOneLaunch.toList}",
        ) &&
        expect(
          afterTwoLaunches.toList == List(110.0f),
          s"a second launch did not compound the update, got ${afterTwoLaunches.toList}",
        )
      }
    }
  }
