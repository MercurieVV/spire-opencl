# Benchmarking

Benchmarks compare the same numeric program as OpenCL, Spire-on-CPU, Breeze, and a plain `while`
loop. These numbers describe one machine, not a portable promise.

## When This Actually Pays Off

OpenCL wins when three things line up: a **large array** (below ~10^6 elements, launch overhead
dominates and the JVM wins outright), **heavy math** (cheap formulas like `a*b+c` can be up to 11x
*slower* on the GPU), and **enough chained work per element** (a single FMA isn't enough to hide a
transfer; tens of chained steps are). Hit all three and the win is an order of magnitude or more —
up to **~195x** for `heavy`, **~688x** for the 4x-heavier `veryHeavy`. Miss them and OpenCL can lose
outright. This is a compute accelerator for compute-bound work, not a blanket "GPU makes everything
faster" claim — see the tables below for exactly where the line sits.

<!-- BENCHMARK_SOURCE -->

## Benchmark Results

### Elementwise, 10^7 floats

Purpose: cheapest possible math, so transfer/launch cost is not hidden by compute. Shows the worst
case for OpenCL.

Tested formula: `elementwise(a, b, c) = a*b + c`.

<!-- BENCHMARK_ELEMENTWISE_TABLE -->

Resident arrays avoid per-launch input upload. Mapped output removes the readback copy on this
unified-memory machine.

### Heavy math, 10^7 floats

Purpose: expensive math (sin/exp/sqrt), same data size as elementwise. Shows the library's actual
sweet spot.

Tested formula: `heavy(x, a) = sqrt(exp(sin(x*a))^2 + sin(x*a)^2)`.

<!-- BENCHMARK_HEAVY_TABLE -->

The GPU wins clearly once the expression has enough arithmetic to amortize launch and transfer.

### Generator, 10^7 floats

Purpose: same `chain`/`heavy` math, but no array upload at all — isolates pure compute cost from
transfer cost.

Tested formulas: `chain(x, a, b, depth) = acc = x; repeat depth times: acc = acc*a + b` (depth 8
here) and `heavy(x, a) = sqrt(exp(sin(x*a))^2 + sin(x*a)^2)`.

<!-- BENCHMARK_GENERATOR_TABLE -->

Generator kernels derive values from `Expr.Index`; there is no input array upload.

### Generator, `veryHeavy` (sin/exp/sqrt composed 4x)

`heavy`'s body run 4 times over (`Bench.VeryHeavyDepth`), same traffic as `heavy`, 4x the
arithmetic. Single-fork quick run, not the 3-fork config the rest of this page uses.

Purpose: check if the OpenCL win keeps growing when compute gets heavier still, at the same array
size.

Tested formula: `veryHeavy(x, a) = heavy(x, a)` composed 4x (sin/exp/sqrt repeated 4 times).

<!-- BENCHMARK_VERYHEAVY_TABLE -->

Device time barely moves between `heavy` and `veryHeavy`; CPU time scales with the added
arithmetic, so the win grows with arithmetic intensity rather than staying fixed.

### Depth sweep, 10^6 floats

Purpose: find how many chained multiply-adds it takes before OpenCL beats the JVM.

Tested formula: `chain(x, a, b, depth) = acc = x; repeat depth times: acc = acc*a + b`.

<!-- BENCHMARK_DEPTHSWEEP_TABLE -->

The OpenCL row is nearly flat: transfer plus launch dominates, while extra multiply-add levels are
cheap on the device.

### Host-side phase split, 10^7 floats

Purpose: see where OpenCL time actually goes — upload, kernel launch, or readback.

Per-launch cost broken into uploading parameters to the device, the kernel launch itself, and
reading the result back. `readback` is not device compute time: the upload and launch enqueue
asynchronously, so almost all device time is absorbed there as the blocking wait.

<!-- BENCHMARK_PHASES_TABLE -->

## What Each Row Actually Runs

Every contender executes the *same* two formulas, defined once in `Programs.scala` and reified for
each backend so the comparison is not GPU-code-vs-hand-tuned-CPU-code:

- **`chain(x, a, b, depth)`** — `acc = x; repeat depth times: acc = acc * a + b`. At `depth = 1`
  this is `a*x + b` (saxpy's inner expression). Memory-bound: FMA only, no transcendentals.
  `depth` is swept so time-vs-depth gives a line whose intercept is transfer + launch latency and
  whose slope is pure arithmetic throughput.
- **`heavy(x, a)`** — `sqrt(exp(sin(x*a))^2 + sin(x*a)^2)`. Compute-bound: same array traffic as
  `chain` at depth 1, but sin/exp/sqrt dominate instead of memory. On the GPU these are hardware
  transcendental units; on the JVM they're `StrictMath` calls.
- **`elementwise(a, b, c)`** — `a*b + c` over three input arrays, one output.

### Encodings (how the array reaches the kernel)

- **generator (encoding A)** — value is a function of `Expr.Index` only; no array is uploaded at
  all, just two scalar uniforms. Best case for the library: `size = N` on the GPU's fast dimension.
- **packed params (encoding B, `elementwise*` rows)** — this library has no native "array
  argument" node, so a,b,c are packed element-major into one buffer and shipped as *batch
  parameters* along dimension 1, while dimension 0 (the fast-varying axis the generated code
  expects) is a single work-item. `openclPacked` starts from data already interleaved (upload only);
  `openclPacking` includes the interleave step itself — the number a caller with three separate
  arrays actually pays.
- **device inputs (encoding C, `openclInput*` rows)** — same `elementwise`/`heavy` arithmetic, but
  arrays live in device memory via `Expr.Input`/`writeInput`, uploaded once and reused, with
  `size = N` on the fast dimension like encoding A. Pairing B and C numbers is exactly the cost of
  the missing array-argument node.
- **mapped output** — device inputs, plus `renderBatchMappedUnsafe` to skip the device→host
  readback copy; result stays in a mapped buffer instead of a fresh host array.

### The JVM/CPU rows

- **Spire** — `Programs.chain`/`heavy` instantiated at `Float` through Spire's `Field`/`Trig`/
  `NRoot` typeclasses — the exact same source the OpenCL kernel is reified from, just run on JVM
  primitives instead of compiled to a kernel.
- **plain while loop** — the same expression hand-transcribed as a raw `while` loop over
  `Array[Float]`, no typeclass dispatch. This is the floor: it's what tells you Spire's abstraction
  overhead (if any) actually costs on the JVM, since without it a ratio between the other rows has
  no scale.
- **Breeze** — `DenseVector[Float]` ops (`a *:* b + c`, `tabulate`). For `heavy`, Breeze has no
  `DenseVector[Float]` sin/exp/sqrt (only `Double`), so that row falls back to a per-element scalar
  map rather than silently promoting precision — reported as the fallback it is, not a fair
  vectorised number.

`AgreementSpec` runs all of these on the same inputs and checks they agree numerically, so "same
formula, different backend" is enforced, not just asserted in a comment.

### Why OpenCL is sometimes slower, not always much faster

- **Fixed cost per launch.** Every OpenCL call pays kernel-launch latency plus (for encodings A/B)
  a data transfer, before a single FLOP runs. At small `size` this fixed cost dominates and a
  `while` loop that never leaves the CPU cache wins outright — see `DepthSweepBench`'s intercept
  (transfer + launch) vs. slope (arithmetic) framing.
- **`chain` is memory-bound, not compute-bound.** One FMA per depth level is nowhere near enough
  arithmetic to hide a PCIe upload/download; the bottleneck is bytes moved, and a modern CPU's
  memory bandwidth for a single float array isn't dramatically worse than the bus transfer to a
  discrete/integrated GPU.
- **Encoding B is deliberately the bad shape.** Packed params put the array on the slow batch
  dimension with dimension 0 (the one the kernel is optimized for) at width 1 — the numbers are
  there to show what NOT having an array-argument node costs, not to flatter the library.
- **`heavy` is where the GPU should win**, since transcendentals are real hardware units there
  and `StrictMath` calls on the JVM. On the recorded Apple M3 Max run it does, decisively: the
  generator-encoding `heavy` row is ~195x faster than the plain while loop at 10^7 elements, and
  even encoding B (packed params, the deliberately bad shape) is still ~9.7x faster. The case where
  OpenCL is *not* much faster is the cheap, memory-bound `elementwise` workload (`a*b+c`) compared
  against the fast JVM rows (Spire/plain) — see the "vs Spire" column in the Elementwise table above.

## Benchmarking Notes

Run correctness first. Timings are meaningless if the implementations do not agree:

```bash
./mill bench.test
```

Then run JMH:

```bash
./mill bench.runJmh GeneratorBench
./mill bench.runJmh -f 1 -wi 2 -i 3 GeneratorBench
```

Record and compare history:

```bash
./mill bench.record -f 1 -wi 2 -i 3 GeneratorBench
./mill bench.history
./mill bench.compare
```

Reports are written to `bench/results/<timestamp>.json`.

Small arrays mostly measure launch latency. Larger arrays show transfer cost and device throughput:

| row | shape |
|---|---|
| generator | values derived from `Expr.Index`; no input array upload |
| packed params | host arrays packed and uploaded every launch |
| device inputs | arrays uploaded with `writeInput`, then reused across launches |
| mapped output | device inputs plus `renderBatchMappedUnsafe` to avoid the readback copy |

Use `./mill bench.report` on a recorded run for rates. Use `./mill bench.precision` to check OpenCL
math accuracy on the current device.
