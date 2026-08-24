package io.github.mercurievv.spireopencl.bench

import io.github.mercurievv.spireopencl.symbolic.{Expr, Formula, Reify, instances}

/** `Programs` reified — the same functions, applied to symbolic arguments instead of floats.
  *
  * Nothing here is benchmark-specific: this is exactly what a caller writes to get a kernel, so the compile step being cheap or expensive is part of
  * what the numbers describe.
  */
object Formulas:

  import instances.given

  /** Encoding **A**, the library's designed shape: `size = N` work-items in dimension 0, one batch element, and the value is a function of the index.
    * Nothing is uploaded per launch but two scalars.
    *
    * `Expr.Index` reaches the kernel as `fi = (float)i`, so above 2^24 the index stops being exactly representable. The largest size benchmarked is
    * 10^7, comfortably under; a larger sweep would have to account for it.
    */
  def generatorChain(depth: Int): Formula =
    Reify(uniforms = List("a", "b"), params = Nil)((uniform, _) => Programs.chain(Expr.Index, uniform("a"), uniform("b"), depth))

  def generatorHeavy: Formula =
    Reify(uniforms = List("a"), params = Nil)((uniform, _) => Programs.heavy(Expr.Index, uniform("a")))

  /** `heavy`'s body composed `depth` times: same traffic as `generatorHeavy` (nothing uploaded but two scalars, one float back), far more arithmetic
    * per element. The row that isolates arithmetic intensity from every other variable this suite controls for.
    */
  def generatorVeryHeavy(depth: Int): Formula =
    Reify(uniforms = List("a"), params = Nil)((uniform, _) => Programs.veryHeavy(Expr.Index, uniform("a"), depth))

  /** Encoding **B**, the array-elementwise fit: `size = 1`, and the array runs along the *batch* dimension, one element per array slot, with the
    * three inputs arriving as per-element parameters.
    *
    * This is the only way to feed host arrays into a kernel that has no array argument. It puts the array on dimension 1 while dimension 0 — the
    * fast-varying one — is a single work-item, which is not the axis the generated code was shaped for. Measuring how much that costs is the point.
    */
  val elementwise: Formula =
    Reify(uniforms = Nil, params = List("a", "b", "c"))((_, param) => Programs.elementwise(param("a"), param("b"), param("c")))

  /** Encoding **C**, the same elementwise work through device-resident input arrays.
    *
    * The arrays live in device memory and are read at the work-item index, so `size = N` on dimension 0 — the same good shape encoding A has — and a
    * launch transfers nothing at all. Where B pays a full upload of every array on every launch and runs a one-work-item-wide grid, C pays the upload
    * once at setup and runs the grid the generated code was written for. It is what B should have been, and the pair of numbers is the measurement of
    * what the `Expr.Input` node bought.
    */
  val elementwiseInputs: Formula =
    Reify.arrays(Nil, Nil, List("a", "b", "c"))((_, _, in) => Programs.elementwise(in("a"), in("b"), in("c")))

  val elementwiseHeavyInputs: Formula =
    Reify.arrays(Nil, Nil, List("a", "b", "c"))((_, _, in) => Expr.mul(Programs.heavy(in("a"), in("b")), in("c")))

  /** Encoding B carrying the compute-bound program, so the elementwise comparison has both an arithmetic-light and an arithmetic-heavy row. */
  val elementwiseHeavy: Formula =
    Reify(uniforms = Nil, params = List("a", "b", "c"))((_, param) => Expr.mul(Programs.heavy(param("a"), param("b")), param("c")))
