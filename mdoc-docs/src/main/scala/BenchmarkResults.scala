package docs

import java.nio.file.Files
import java.nio.file.Path
import java.nio.charset.StandardCharsets
import upickle.default.*

object BenchmarkResults:

  private final case class Row(benchmark: String, params: Map[String, String], score: Double, error: Double)

  /** Per-launch host-side phase split, from the `PhaseCounters` aux counters `ClKernel.launch` emits: nanosecond totals over an iteration plus the
    * launch count to divide by. `readback` is not device time — the upload and NDRange enqueue are non-blocking, so almost all device time lands
    * there as the blocking wait, not as processing.
    */
  private final case class Phases(uploadUs: Double, launchUs: Double, readbackUs: Double)

  /** Every arg but the last is a JSON results file; the last is the markdown to patch. Multiple files let a full multi-fork run and a smaller,
    * targeted extension (different date, different config) contribute to the same table — `find`/`rows` below take whichever file's row matches
    * first, so an extension file should introduce new benchmark/param combinations rather than override the main run's.
    */
  def main(args: Array[String]): Unit =
    val jsonPaths = args.init.map(Path.of(_)).toSeq
    val markdownPath = Path.of(args.last)
    val raw = jsonPaths.flatMap(jsonPath => read[ujson.Value](Files.readString(jsonPath)).arr.toSeq)
    val rows = raw.map { value =>
      val obj = value.obj
      val metric = obj("primaryMetric").obj
      val params = obj.get("params").fold(Map.empty[String, String])(_.obj.view.mapValues(_.str).toMap)
      Row(
        obj("benchmark").str.split('.').takeRight(2).mkString("."),
        params,
        metric("score").num,
        metric("scoreError").num,
      )
    }
    val phases = raw.flatMap { value =>
      val obj = value.obj
      val name = obj("benchmark").str.split('.').takeRight(2).mkString(".")
      val params = obj.get("params").fold(Map.empty[String, String])(_.obj.view.mapValues(_.str).toMap)
      for
        sm       <- obj.get("secondaryMetrics").map(_.obj)
        upload   <- sm.get("uploadNs")
        launch   <- sm.get("launchNs")
        readback <- sm.get("readbackNs")
        ops      <- sm.get("ops")
      yield (name, params) -> Phases(
        upload("score").num / ops("score").num / 1e3,
        launch("score").num / ops("score").num / 1e3,
        readback("score").num / ops("score").num / 1e3,
      )
    }.toMap

    val markdown = Files.readString(markdownPath)
    val generated = render(jsonPaths.map(_.getFileName.toString).mkString(", "), rows, phases)
    Files.writeString(
      markdownPath,
      markdown.replace("<!-- BENCHMARK_RESULTS -->", generated),
      StandardCharsets.UTF_8,
    )

  private def render(sourceName: String, rows: Seq[Row], phases: Map[(String, Map[String, String]), Phases]): String =
    List(
      s"Generated from `bench/results/$sourceName`. Scores are `us/op`, lower is better. `+/-` is JMH's 99.9% confidence half-width.",
      "",
      "## Benchmark Results",
      "",
      "### Elementwise, 10^7 floats",
      "",
      table(
        Seq("row", "us/op", "vs Spire"),
        Seq(
          entry(
            rows,
            "ElementwiseBench.openclInputMapped",
            "10000000",
            "OpenCL, resident arrays + mapped output",
            fasterThan(
              rows,
              "ElementwiseBench.spireElementwise",
              "10000000",
              rows.find(matchRow("ElementwiseBench.openclInputMapped", "10000000")).get.score,
            ),
          ),
          entry(
            rows,
            "ElementwiseBench.openclInput",
            "10000000",
            "OpenCL, resident arrays",
            slowerThan(
              rows,
              "ElementwiseBench.spireElementwise",
              "10000000",
              rows.find(matchRow("ElementwiseBench.openclInput", "10000000")).get.score,
            ),
          ),
          entry(rows, "ElementwiseBench.spireElementwise", "10000000", "Spire on JVM", "1.0"),
          entry(rows, "ElementwiseBench.plainElementwise", "10000000", "plain while loop", "-"),
          entry(
            rows,
            "ElementwiseBench.breezeElementwise",
            "10000000",
            "Breeze DenseVector",
            slowerThan(
              rows,
              "ElementwiseBench.spireElementwise",
              "10000000",
              rows.find(matchRow("ElementwiseBench.breezeElementwise", "10000000")).get.score,
            ),
          ),
          entry(
            rows,
            "ElementwiseBench.openclPacked",
            "10000000",
            "OpenCL, packed params",
            slowerThan(
              rows,
              "ElementwiseBench.spireElementwise",
              "10000000",
              rows.find(matchRow("ElementwiseBench.openclPacked", "10000000")).get.score,
            ),
          ),
          entry(
            rows,
            "ElementwiseBench.openclPacking",
            "10000000",
            "OpenCL, packed params + interleave",
            slowerThan(
              rows,
              "ElementwiseBench.spireElementwise",
              "10000000",
              rows.find(matchRow("ElementwiseBench.openclPacking", "10000000")).get.score,
            ),
          ),
        ),
      ),
      "",
      "Resident arrays avoid per-launch input upload. Mapped output removes the readback copy on this unified-memory machine.",
      "",
      "### Heavy math, 10^7 floats",
      "",
      table(
        Seq("row", "us/op"),
        Seq(
          entry(rows, "ElementwiseBench.openclInputHeavy", "10000000", "OpenCL, resident arrays"),
          entry(rows, "ElementwiseBench.openclHeavy", "10000000", "OpenCL, packed params"),
          entry(rows, "ElementwiseBench.spireHeavy", "10000000", "Spire on JVM"),
          entry(rows, "ElementwiseBench.plainHeavy", "10000000", "plain while loop"),
          entry(rows, "ElementwiseBench.breezeHeavy", "10000000", "Breeze scalar fallback"),
        ),
      ),
      "",
      "The GPU wins clearly once the expression has enough arithmetic to amortize launch and transfer.",
      "",
      "### Generator, 10^7 floats",
      "",
      table(
        Seq("row", "chain depth 8", "sin/exp/sqrt"),
        Seq(
          Seq(
            "OpenCL",
            fmt(find(rows, "GeneratorBench.openclChain", "10000000")),
            fmt(find(rows, "GeneratorBench.openclHeavy", "10000000")),
          ),
          Seq(
            "plain while loop",
            fmt(find(rows, "GeneratorBench.plainChain", "10000000")),
            fmt(find(rows, "GeneratorBench.plainHeavy", "10000000")),
          ),
          Seq(
            "Spire on JVM",
            fmt(find(rows, "GeneratorBench.spireChain", "10000000")),
            fmt(find(rows, "GeneratorBench.spireHeavy", "10000000")),
          ),
          Seq("Breeze", fmt(find(rows, "GeneratorBench.breezeChain", "10000000")), "n/a"),
        ),
      ),
      "",
      "Generator kernels derive values from `Expr.Index`; there is no input array upload.",
      "",
      "### Generator, `veryHeavy` (sin/exp/sqrt composed 4x)",
      "",
      "`heavy`'s body run 4 times over (`Bench.VeryHeavyDepth`), same traffic as `heavy`, 4x the " +
        "arithmetic. Single-fork quick run, not the 3-fork config the rest of this page uses.",
      "",
      table(
        Seq("size", "OpenCL", "plain while loop", "Spire on JVM"),
        Seq("10000", "1000000", "10000000").map { size =>
          Seq(
            size,
            fmt(find(rows, "GeneratorBench.openclVeryHeavy", size)),
            fmt(find(rows, "GeneratorBench.plainVeryHeavy", size)),
            fmt(find(rows, "GeneratorBench.spireVeryHeavy", size)),
          )
        },
      ),
      "",
      "Device time barely moves between `heavy` and `veryHeavy`; CPU time scales with the added " +
        "arithmetic, so the win grows with arithmetic intensity rather than staying fixed.",
      "",
      "### Depth sweep, 10^6 floats",
      "",
      table(
        Seq("depth", "OpenCL", "plain", "Spire"),
        Seq("1", "8", "32", "128").map { depth =>
          Seq(
            depth,
            fmt(find(rows, "DepthSweepBench.opencl", depth)),
            fmt(find(rows, "DepthSweepBench.plain", depth)),
            fmt(find(rows, "DepthSweepBench.spire", depth)),
          )
        },
      ),
      "",
      "The OpenCL row is nearly flat: transfer plus launch dominates, while extra multiply-add levels are cheap on the device.",
      "",
      "### Host-side phase split, 10^7 floats",
      "",
      "Per-launch cost broken into uploading parameters to the device, the kernel launch itself, and reading the result back. `readback` is not " +
        "device compute time: the upload and launch enqueue asynchronously, so almost all device time is absorbed there as the blocking wait.",
      "",
      table(
        Seq("row", "upload us/op", "launch us/op", "readback us/op"),
        Seq(
          phaseEntry(phases, "ElementwiseBench.openclInput", "10000000", "OpenCL, resident arrays"),
          phaseEntry(
            phases,
            "ElementwiseBench.openclInputMapped",
            "10000000",
            "OpenCL, resident arrays + mapped output",
          ),
          phaseEntry(phases, "ElementwiseBench.openclPacked", "10000000", "OpenCL, packed params"),
          phaseEntry(
            phases,
            "ElementwiseBench.openclPacking",
            "10000000",
            "OpenCL, packed params + interleave",
          ),
          phaseEntry(phases, "GeneratorBench.openclChain", "10000000", "Generator, chain depth 8"),
        ),
      ),
    ).mkString("\n")

  private def table(header: Seq[String], rows: Seq[Seq[String]]): String =
    (Seq(header, header.map(_ => "---")) ++ rows).map(_.mkString("| ", " | ", " |")).mkString("\n")

  private def matchRow(benchmark: String, param: String)(row: Row): Boolean =
    row.benchmark == benchmark && row.params.values.exists(_ == param)

  private def phaseEntry(
    phases: Map[(String, Map[String, String]), Phases],
    benchmark: String,
    param: String,
    label: String,
  ): Seq[String] =
    val found = phases
      .collectFirst { case ((name, params), p) if name == benchmark && params.values.exists(_ == param) => p }
      .getOrElse(sys.error(s"missing phase counters for: $benchmark / $param"))
    Seq(label, round(found.uploadUs), round(found.launchUs), round(found.readbackUs))

  private def find(rows: Seq[Row], benchmark: String, param: String): Row =
    rows.find(matchRow(benchmark, param)).getOrElse(sys.error(s"missing benchmark result: $benchmark / $param"))

  private def entry(
    rows: Seq[Row],
    benchmark: String,
    param: String,
    label: String,
    extra: String,
  ): Seq[String] =
    Seq(label, fmt(find(rows, benchmark, param)), extra)

  private def entry(rows: Seq[Row], benchmark: String, param: String, label: String): Seq[String] =
    Seq(label, fmt(find(rows, benchmark, param)))

  private def fmt(row: Row): String =
    s"${round(row.score)} +/- ${round(row.error)}"

  private def round(value: Double): String =
    if value >= 100 then f"$value%.0f" else if value >= 10 then f"$value%.1f" else f"$value%.2f"

  private def slowerThan(rows: Seq[Row], baseBenchmark: String, param: String, score: Double): String =
    f"${score / find(rows, baseBenchmark, param).score}%.1fx slower"

  private def fasterThan(rows: Seq[Row], baseBenchmark: String, param: String, score: Double): String =
    f"${find(rows, baseBenchmark, param).score / score}%.1fx faster"
