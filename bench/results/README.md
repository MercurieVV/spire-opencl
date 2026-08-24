# Recorded runs

JMH JSON, one file per run, named by date and device. Produced by

```bash
./mill bench.runJmh -rf json -rff bench/results/<date>-<device>.json
```

Each file holds every iteration of every fork (`primaryMetric.rawData`), the confidence interval and
percentiles, the `uploadNs` / `launchNs` / `readbackNs` / `ops` aux counters, and the JVM and JVM-args
the run used — enough to re-derive any summary below, or to notice that one fork disagreed with the
other two.

These are kept because the numbers quoted in `docs/benchmarking.md` and in the commit messages need a
source. They describe **one machine** and are not a portable claim: a discrete GPU changes the
transfer costs, and `renderBatchMappedUnsafe` in particular is fast here because Apple Silicon shares
physical memory with the host.

---

## 2026-08-15, Apple M3 Max

`jdk 21.0.10`, 3 forks, 5×1s warmup, 10×1s measurement, `-Xms8g -Xmx8g -XX:+AlwaysPreTouch
-XX:MaxDirectMemorySize=2g`. All figures µs/op, lower is better; `±` is the 99.9% confidence
half-width.

### Elementwise `a * b + c`, 10^7 floats

| | µs | vs Spire |
|---|---|---|
| opencl, packed params + interleave | 17033 ± 482 | 11.2x slower |
| opencl, packed params (pre-interleaved) | 10372 ± 303 | 6.8x slower |
| Breeze `DenseVector[Float]` | 4250 ± 100 | 2.8x slower |
| opencl, device-resident arrays | 1923 ± 78 | 1.26x slower |
| plain `while` loop | 1570 ± 38 | — |
| Spire on JVM | 1523 ± 24 | 1.0 |
| **opencl, resident + mapped readback** | **1085 ± 95** | **1.4x faster** |

The first configuration in which the GPU beats the JVM on a purely memory-bound kernel.

### Compute-bound (`sin`, `exp`, `sqrt`), 10^7 floats

| | µs |
|---|---|
| Breeze — scalar fallback, no `DenseVector[Float]` ufuncs | 153269 ± 1801 |
| plain `while` loop | 110843 ± 450 |
| Spire on JVM | 110101 ± 620 |
| opencl, packed params | 11413 ± 467 |
| opencl, device-resident arrays | 3573 ± 804 |

The resident row is the one noisy result in the set — a single-fork run of the same benchmark gave
2144 µs, so the fork-to-fork spread is real and 3573 should be read as an upper bound.

### Generator (a function of the index, no input at all), 10^7 floats

| | chain, depth 8 | sin/exp/sqrt |
|---|---|---|
| opencl | 1752 ± 122 | 2076 ± 158 |
| plain `while` loop | 21975 ± 147 | 404512 ± 1682 |
| Spire on JVM | 80627 ± 5592 | 404414 ± 1557 |
| Breeze | 100750 ± 1479 | n/a |

## 2026-08-24, Apple M3 Max — arithmetic intensity extension

`GeneratorBench`, `-f 1 -wi 2 -i 3` (1 fork, lower confidence than the run above), `2026-08-24-170344.json`.
`veryHeavy` = `heavy`'s body composed `Bench.VeryHeavyDepth` (4) times, same traffic as `heavy`.

| size | opencl | plain while loop | vs plain |
|---|---|---|---|
| 10^4 | 170.6 us | 1036.8 us | **6.1x faster** |
| 10^6 | 364.3 us | 105429.4 us | **289.4x faster** |
| 10^7 | 2012.5 us | 1383969.2 us | **687.7x faster** |

### Depth sweep at 10^6 — separating compute from transfer

| depth | opencl | plain | Spire |
|---|---|---|---|
| 1 | 346 ± 4 | 815 ± 20 | 2559 ± 49 |
| 8 | 355 ± 5 | 2362 ± 175 | 7474 ± 47 |
| 32 | 387 ± 10 | 11371 ± 56 | 19931 ± 1296 |
| 128 | 383 ± 3 | 109889 ± 1379 | 131254 ± 1199 |

Slope ≈ **0.3 µs per multiply-add level** on the device against ≈ 858 µs on the CPU, with a 345 µs
intercept. Arithmetic is effectively free; the intercept — transfer plus launch latency — is the
whole cost.

### Where a launch's time goes, 10^7 elementwise

| | upload | launch | readback |
|---|---|---|---|
| packed params | 5008 | 8 | 5336 |
| device-resident arrays | 0.4 | 9 | 1907 |
| resident + mapped | 0.1 | 9 | 1058 |

Host-side enqueue costs, not kernel time: the upload is enqueued `CL_FALSE` and the NDRange
asynchronously, so device time is absorbed by the blocking readback. Read together with the depth
sweep, which is what actually separates the two.

### Crossover

Launch latency floors the GPU at ~180 µs, so at 10^4 every JVM row wins — Spire does elementwise in
0.9 µs. The memory-bound crossover is around 5×10^6. Compute-bound, the GPU is already 27x ahead at
10^6.
