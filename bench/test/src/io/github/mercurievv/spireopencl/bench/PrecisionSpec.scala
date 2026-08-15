package io.github.mercurievv.spireopencl.bench

import cats.effect.IO
import io.github.mercurievv.spireopencl.symbolic.UnOp
import weaver.*

/** The accuracy guard.
  *
  * `bench.precision` prints the whole picture; this asserts the part that must not change. A guard nobody runs is not a guard, so the bounds live in
  * the suite that runs on every `bench.test` rather than only in a report someone has to remember to look at.
  *
  * What it protects against is specific: a change to `CodeGen` that keeps every answer plausible and loses several bits. No timing would show it, and
  * the agreement tests use a 1e-4 relative tolerance that would swallow it whole.
  */
object PrecisionSpec extends SimpleIOSuite:

  /** See `KernelSpec`: concurrent OpenCL context creation crashes the driver. */
  override def maxParallelism: Int = 1

  test("every transcendental is within the ULP the OpenCL specification permits") {
    IO {
      val over = UnOp.values.toList
        .filter(Precision.domains.contains)
        .map(op => op -> Precision.measure(op, samples = 1 << 12))
        .filter((op, s) => !s.within(Precision.allowed(op)))
      if over.isEmpty then success
      else
        failure(
          over.map((op, s) => s"$op: ${s.max} ULP > ${Precision.allowed(op)} allowed, worst at ${s.worstInput}").mkString("; "),
        )
    }
  }

  test("ULP distance is a count of representable floats, in both directions and across zero") {
    IO {
      expect(Precision.ulps(1.0f, 1.0f) == 0L) and
        expect(Precision.ulps(1.0f, Math.nextUp(1.0f)) == 1L) and
        expect(Precision.ulps(Math.nextUp(1.0f), 1.0f) == 1L) and
        expect(Precision.ulps(-0.0f, 0.0f) == 0L) and
        expect(Precision.ulps(Float.NaN, 1.0f) == Long.MaxValue)
    }
  }

  /** Not an assertion about which way it goes — both are legal, and a discrete GPU may well differ from this one. It runs so that the answer is on
    * the record for whatever device the suite is run on, since an algorithm relying on gradual underflow needs to know before it runs, not after.
    */
  test("subnormal handling is determined rather than assumed") {
    IO {
      val flushed = Precision.flushesSubnormals
      println(s"  subnormals ${if flushed then "flush to zero" else "survive"} on this device")
      success
    }
  }
