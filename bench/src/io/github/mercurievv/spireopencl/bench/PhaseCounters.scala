package io.github.mercurievv.spireopencl.bench

import org.openjdk.jmh.annotations.*

/** Where a launch's wall clock falls, reported as JMH secondary metrics.
  *
  * Read these as **host-side** costs, never as kernel time. `ClKernel` enqueues the parameter upload with `CL_FALSE` and the NDRange asynchronously,
  * so "upload" and "launch" measure the driver call returning, not the work finishing; almost all device time is absorbed by the blocking
  * `clEnqueueReadBuffer` and therefore lands in "readback". A genuine device-only figure needs `CL_QUEUE_PROFILING_ENABLE` and
  * `clGetEventProfilingInfo`, which the library does not expose.
  *
  * The compute/transfer split these counters cannot give is recovered instead by `DepthSweepBench`, which varies arithmetic at a fixed transfer size:
  * slope is compute, intercept is transfer plus launch latency.
  *
  * JMH turns every public no-argument method of an `@AuxCounters` state into a counter, so this class exposes exactly three — the Scala accessors for
  * the three `var`s — and nothing else. `begin` and `record` take arguments precisely so they are not mistaken for counters.
  */
/* `EVENTS`, so each counter is the raw nanosecond total over the iteration. `OPERATIONS` looks like the right choice — it claims to divide by the
 * operation count — but JMH then also runs the result through the benchmark's own time-unit conversion, and the printed numbers come out scaled by a
 * factor that does not correspond to anything. The `ops` counter below makes the division explicit instead: `uploadNs / ops` is nanoseconds per
 * launch, and the three phases plus the unaccounted remainder sum to the primary score. */
@AuxCounters(AuxCounters.Type.EVENTS)
@State(Scope.Thread)
class PhaseCounters:
  var uploadNs: Long = 0L
  var launchNs: Long = 0L
  var readbackNs: Long = 0L

  /** Launches this iteration — the divisor that turns the three totals into per-launch nanoseconds. */
  var ops: Long = 0L

  private var start: Long = 0L
  private var upload: Long = 0L
  private var launch: Long = 0L

  def begin(now: Long): Unit = start = now

  def record(phase: String, at: Long): Unit =
    phase match
      case "upload"   => upload = at
      case "launch"   => launch = at
      case "readback" =>
        uploadNs   += upload - start
        launchNs   += launch - upload
        readbackNs += at - launch
        ops        += 1L
      case _ => ()

  @Setup(Level.Iteration)
  def reset(): Unit =
    uploadNs   = 0L
    launchNs   = 0L
    readbackNs = 0L
    ops        = 0L
