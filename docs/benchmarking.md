# Benchmarking

Benchmarks compare the same numeric program as OpenCL, Spire-on-CPU, Breeze, and a plain `while`
loop. These numbers describe one machine, not a portable promise.

## When This Actually Pays Off

OpenCL wins big only when three things hold together: **large arrays** (below ~10^6 elements, the
~180 us launch floor dominates and the JVM wins outright), **arithmetic-heavy formulas** (cheap ones
like `a*b+c` can't hide even a same-machine copy — best case ~29% faster than Spire, worst case 11x
*slower*), and **deep chains** (`chain` needs depth in the tens before the device pulls ahead; the
depth sweep shows OpenCL flat from depth 1 to 128 while JVM scales linearly).

Where all three line up — large N, compute-bound (`heavy`: `sin`/`exp`/`sqrt`) — the device wins by
an order of magnitude or more: ~195x at 10^7 elements, ~27x at 10^6. That is the library's actual
niche, not a general "GPU makes everything faster" claim.

`veryHeavy` composes `heavy`'s transcendentals 4x, at the same array traffic
(`bench/results/2026-08-24-170344.json`, quick single-fork run). At 10^7 elements the win grows to
**~688x**; at 10^4 elements, where `heavy` itself still lost, OpenCL now wins **~6.1x**. The
small-array floor moves with arithmetic intensity, not just with size. Full table in
`bench/results/README.md`.

<!-- BENCHMARK_RESULTS -->

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
  against the fast JVM rows (Spire/plain), not `heavy` — see the percent table below.

### GPU vs. CPU win, in percent

`Nx faster/slower` reads fine for order-of-magnitude gaps but is a bad unit close to parity — "1.4x
faster" and "40% faster" are the same fact stated two ways, and percent is the more legible one when
the two numbers are close. So: percent where the comparison is near 1x, `Nx` (as in the generated
results table above) where it's a large multiple and a percent would just be an ugly four-digit
number. Both are the *same underlying numbers* from `bench/results/2026-08-15-apple-m3-max.json`,
just two units for two shapes of gap.

**Elementwise `a*b+c`, 10^7 floats — near parity, percent is the useful unit:**

| row | vs Spire on JVM (1523 us) |
|---|---|
| OpenCL, resident arrays + mapped output (1085 us) | **29% less time** (faster) |
| plain while loop (1570 us) | 3% more time |
| OpenCL, resident arrays (1923 us) | 26% more time |
| Breeze `DenseVector` (4250 us) | 179% more time |

**Order-of-magnitude gaps — kept as `Nx`, not percent, on purpose:**

| workload | OpenCL row | vs plain while loop |
|---|---|---|
| `chain` depth 8 (generator) | 1752 us | ~12.5x faster |
| `heavy` (generator) | 2076 us | ~195x faster |
| `heavy` (encoding C, resident arrays) | 3573 us | ~31x faster |
| `heavy` (encoding B, packed params) | 11413 us | ~9.7x faster |

Converting the last row to percent would read "870% faster" — technically correct, harder to parse
at a glance than "9.7x," which is why the generated table above sticks to `Nx` there.
