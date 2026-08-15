# Benchmarking

JMH benchmarks comparing this library against the same numeric program on the JVM — Spire
typeclasses over primitive arrays, Breeze `DenseVector`, and a plain `while` loop as the floor.

The module is `bench`; sources live under `bench/src`.

## Running

```bash
./mill bench.test                      # correctness gate — run this first
./mill bench.runJmh                    # everything (~40 min)
```

`bench.test` is not optional. The four contenders must compute the same thing for any timing to
mean anything, and nothing about a timing reveals that they do not — a GPU row being four times
faster looks identical whether it is faster or whether it is evaluating a shorter expression.

## Keeping history

`bench.runJmh` runs and forgets. `bench.record` runs, keeps the report, and compares it against the
previous one:

```bash
./mill bench.record                          # full matrix, recorded, compared against the last run
./mill bench.record -f 1 -wi 2 -i 3 GeneratorBench   # any runJmh argument still works
./mill bench.history                         # what has been recorded
./mill bench.compare                         # the two most recent, without running anything
./mill bench.compare a.json b.json 5         # two named reports, 5% threshold
```

Reports land in `bench/results/<timestamp>.json` and are meant to be committed — see
`bench/results/README.md`. Neither JMH nor Mill has any notion of a previous run, so this is local
rather than the usual `benchmark-action/github-action-benchmark`, which keeps history on a CI branch
and would be measuring a GPU-less runner here.

**What counts as a regression.** Not simply a slower number. A row is reported only when it is worse
by more than the threshold (10% by default) **and** the two runs' 99.9% confidence intervals do not
overlap — so the claim is that the difference exceeds the spread each run measured for itself, rather
than exceeding an arbitrary percentage. A quick `-f 1 -wi 1 -i 2` run has intervals wide enough that
almost nothing will be flagged, which is correct: that run does not know enough to accuse anything.
`bench.record` and `bench.compare` fail the build when something is flagged.

Benchmarks in the baseline that a filtered run did not measure are counted, not listed — running a
subset is the normal case and printing every excluded row as if deleted would bury the ones measured.

Narrower runs, by regex and parameter:

```bash
./mill bench.runJmh GeneratorBench                       # one class
./mill bench.runJmh -p size=1000000 'ElementwiseBench.*' # one size
./mill bench.runJmh -f 1 -wi 2 -i 3 GeneratorBench       # quick, ~1 min, indicative only
./mill bench.runJmh -rf json -rff bench-results.json     # capture results
./mill bench.runJmh -prof gc GeneratorBench              # confirm the measured path allocates nothing
```

Check the device first, since which one you get is not obvious:

```bash
./mill spireOpencl.runMain io.github.mercurievv.spireopencl.opencl.DeviceProbe
```

Not wired into CI. The GitHub runners have no GPU and use pocl on a shared vCPU; numbers from there
would describe the runner, not the library, and publishing them would be worse than publishing
none.

## What is measured

Everything runs at `Float`. `CodeGen` emits single precision throughout — Apple Silicon reports no
`cl_khr_fp64` and a `double` kernel does not build there — so a `Double` comparison would have no
GPU side at all.

Two workloads, each written once in `Programs.scala` and polymorphic in its value type. Instantiated
at `Expr` it becomes the kernel; instantiated at `Float` it becomes the Spire row. That is the
library's whole claim, and it also removes the usual way this kind of benchmark goes wrong.

- **chain** — a dependent multiply-add chain, `depth` levels deep. Memory-bound at depth 1.
- **heavy** — `sin`, `exp`, `sqrt`. Same bytes moved, far more arithmetic per element.

### Two encodings

`spire-opencl` has **no array kernel argument**. A kernel's inputs are scalar uniforms, one float per
*batch element*, per-element state, and the work-item index. `CodeGen` emits
`out[e * n + i] = f(i, params_e)`. That gives two ways to put a big array through it, and both are
benchmarked:

`spire-opencl` has **three** ways to put a big array through a kernel, and all three are benchmarked:

| | **A — generator** | **B — packed params** | **C — device-resident inputs** |
|---|---|---|---|
| shape | `size = N`, one batch element | `size = 1`, `maxBatch = N` | `size = N`, one batch element |
| the array lives on | dimension 0 | dimension 1 | dimension 0 |
| input | none; a function of the index | three host arrays, interleaved and uploaded **every launch** | three device arrays, uploaded **once** |
| IR node | `Expr.Index` | `Expr.Param` | `Expr.Input` |
| entry point | `renderBatchIntoUnsafe` | `renderBatchPackedUnsafe` | `writeInput` once, then `renderBatchIntoUnsafe` |

**B** is what the library could express before `Expr.Input` existed. It puts the array along
dimension 1 while dimension 0 — the fast-varying one — holds a single work-item, and it makes
`ClKernel` allocate a `3 * N` float staging buffer and a matching device buffer: 120 MB apiece at
10^7, re-sent on every launch. It is kept as a benchmark row because it is the right shape for data
that genuinely changes every launch, and because it is the baseline C is measured against.

B is benchmarked twice: `openclPacked` starts from already-interleaved data, `openclPacking`
includes the interleave. A caller holding three parallel arrays pays the second number.

**C** also has a second row, `openclInputMapped`, which removes the readback copy as well:
`renderBatchMappedUnsafe` asks the driver for a pointer to the results where they already are instead
of copying them into caller memory. It needs `hostVisibleOutput = true` at compile. It is paired with
`openclInput` and deliberately symmetric with it — both launch and leave the caller holding a buffer,
neither reads through it — so the difference between the two rows is the copy and nothing else.

**C** is the shape ordinary array code wants. `Expr.Input(name)` compiles to a `__global const
float*` argument read at `a[i]`, so the array sits in device memory across launches and the grid is
`N` work-items wide — the same shape A has. The upload happens once, outside the measured path,
which is exactly the claim: data that does not change between launches should not be sent with
every launch.

### No reduction row

`Expr.Sum` reduces over the **batch** dimension, not over the array: `CodeGen` emits `acc[e]`, a
barrier, then a serial `for (k < ne)` in one work-item, and `ClKernel` rejects a batch larger than
the device's maximum work-group size. A dot product over 10^7 elements is not expressible, and
timing a 256-element serial batch sum against a 10^7 Breeze dot product would compare nothing.

### Breeze's gaps

`breeze.numerics.sin` / `exp` / `sqrt` have `DenseVector[Double]` implementations, not
`DenseVector[Float]`, so the **heavy** workload has no vectorised Breeze form at this precision and
its row falls back to a per-element `tabulate` — reported as the scalar fallback it is rather than
silently promoted to `Double`, which would be a different computation.

`DenseVector.tabulate` also has no in-place form, so the Breeze generator row allocates a fresh
vector per call while every other row writes into a reused array. That is Breeze's API, not harness
sloppiness, and the allocation is part of what the idiom costs.

## Beyond duration

A duration only answers "how long", which is useful against a decision and nothing else. Four other
questions are measured, each because a duration cannot answer it.

**Rates** — `./mill bench.report`. `Traffic` states what each benchmark moves and how many operations
it performs (counted off the `Expr` DAG after CSE, so it is what the kernel really executes), and
`Report` turns a recorded run into elements/s, GB/s, operations/s and arithmetic intensity. The
ceiling is the best bandwidth observed in that run, taken only from rows whose working set exceeds
64 MB — a 10^4-element row answers out of cache at a rate main memory cannot sustain, and using it
would make every honest row look like a fraction of a number never available to it. Transcendentals
count as one operation, so compute-bound rows read as a lower bound.

**Accuracy** — `./mill bench.precision`, and asserted in `bench.test`. A kernel that is fifty times
faster and quietly less accurate is not fifty times better, and no timing says which one you have.
OpenCL does not promise correctly rounded transcendentals — 4 ULP for `sin`, 3 for `exp`, 16 for
`pow` — so the error is a documented quantity to check against. `Precision` measures ULP error
against each operation's own `Double` definition, which is the definition `CodeGen` claims to compile
faithfully. It also reports error along a dependent chain, and whether the device flushes subnormals
to zero — a cliff rather than a rounding error, and invisible to any speed benchmark.

**Break-even** — `ResidencyBench`. Every other elementwise row takes one of two extremes: upload
every launch, or upload once and never count it. The real question is a ratio — data arrives, then
some number of kernels run before it changes. Both rows carry the upload, and the packed row gets
pre-interleaved data (its best case), so the break-even reported is conservative against residency.

**Marshalling** — `MarshallingBench`. Every other benchmark starts from an `Array[Float]`, which is a
convenient fiction: data arrives as `Double`, or boxed, or row-major out of a columnar store. That
conversion is a full pass, is paid before the library is called at all, and can cost more than the
entire computation.

**Latency against throughput** — `PipelineBench`. A blocking entry point costs a host round trip, and
at small sizes that round trip is not part of the cost but all of it. `pipelined` enqueues the
launches *and their readbacks* and waits once. Enqueuing launches while still collecting them with
blocking reads moves the round trips rather than removing them — the first version of this benchmark
did exactly that and measured no gain at all.

## Separating transfer from compute

`Kernel.onPhase` reports three timestamps per launch and `PhaseCounters` surfaces them as JMH
secondary metrics — but read them as **host-side** costs, never as kernel time. The parameter upload
is enqueued `CL_FALSE` and the NDRange asynchronously, so `uploadNs` and `launchNs` measure driver
calls returning; nearly all device time is absorbed by the blocking `clEnqueueReadBuffer` and lands
in `readbackNs`. A genuine device-only figure needs `CL_QUEUE_PROFILING_ENABLE` and
`clGetEventProfilingInfo`, which the library does not expose.

The counters are raw nanosecond totals per iteration, alongside an `ops` count: divide to get
nanoseconds per launch. (`@AuxCounters(OPERATIONS)` claims to do that division itself, but JMH then
also applies the benchmark's time-unit conversion and the printed numbers come out scaled by nothing
meaningful.)

The split the counters cannot give comes from `DepthSweepBench` instead. It fixes the array at 10^6
and varies arithmetic depth, so every point moves identical bytes and differs only in multiply-adds
per element. Plot time against depth: **intercept is transfer plus launch latency, slope is
arithmetic**. Run the same sweep across the JVM rows and the ratio of slopes is the throughput
ratio with all fixed overhead divided out.

## Harness notes

- `@Threads(1)` throughout: `ClKernel` is explicitly not thread-safe — `clSetKernelArg` mutates the
  kernel object.
- Kernels are compiled in `@Setup(Level.Trial)` via `Resource.allocated` and released in
  `@TearDown`, and live in their own state classes so a Spire or Breeze trial never creates an
  OpenCL context.
- Inputs come from a fixed-seed xorshift, bounded to `[-1, 1)` so `exp` cannot push the JVM rows
  into a denormal benchmark, and output arrays are reused across invocations — at 10^7 a fresh array
  per call would make every JVM row a measurement of the allocator.
- The measured methods call the `*Unsafe` entry points, so no `IO` scheduling enters the timing.
- `-Xms8g -Xmx8g -XX:+AlwaysPreTouch -XX:MaxDirectMemorySize=2g`, the last because of encoding B's
  direct staging buffer.

## Reading the results

Two things are worth checking before believing any of it: at 10^4 the GPU should lose to every JVM
row on launch latency alone, and encoding B should trail encoding A. If either does not hold,
suspect the harness first.
