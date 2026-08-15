# Benchmarking

Benchmarks compare the same numeric program as OpenCL, Spire-on-CPU, Breeze, and a plain `while`
loop. These numbers describe one machine, not a portable promise.

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
