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
    val replaced = sections(jsonPaths.map(_.getFileName.toString).mkString(", "), rows, phases)
      .foldLeft(markdown) { case (md, (marker, content)) => md.replace(marker, content) }
    Files.writeString(markdownPath, replaced, StandardCharsets.UTF_8)

  /** One marker per table, filled in place — headings, formula descriptions, and prose live directly in the markdown source, not here, so they can be
    * edited without touching this file.
    */
  private def sections(
    sourceName: String,
    rows: Seq[Row],
    phases: Map[(String, Map[String, String]), Phases],
  ): Seq[(String, String)] =
    Seq(
      "<!-- BENCHMARK_SOURCE -->" ->
        s"Generated from `bench/results/$sourceName`. Scores are `us/op`, lower is better. `+/-` is JMH's 99.9% confidence half-width.",
      "<!-- BENCHMARK_ELEMENTWISE_TABLE -->" -> table(
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
      "<!-- BENCHMARK_HEAVY_TABLE -->" -> table(
        Seq("row", "us/op", "vs Spire"),
        Seq(
          entryNx(
            rows,
            "ElementwiseBench.openclInputHeavy",
            "10000000",
            "OpenCL, resident arrays",
            "ElementwiseBench.spireHeavy",
          ),
          entryNx(
            rows,
            "ElementwiseBench.openclHeavy",
            "10000000",
            "OpenCL, packed params",
            "ElementwiseBench.spireHeavy",
          ),
          entry(rows, "ElementwiseBench.spireHeavy", "10000000", "Spire on JVM", "1.0"),
          entryNx(
            rows,
            "ElementwiseBench.plainHeavy",
            "10000000",
            "plain while loop",
            "ElementwiseBench.spireHeavy",
          ),
          entryNx(
            rows,
            "ElementwiseBench.breezeHeavy",
            "10000000",
            "Breeze scalar fallback",
            "ElementwiseBench.spireHeavy",
          ),
        ),
      ),
      "<!-- BENCHMARK_GENERATOR_TABLE -->" -> table(
        Seq("row", "chain depth 8", "sin/exp/sqrt", "chain vs Spire", "sin/exp/sqrt vs Spire"),
        Seq(
          {
            val chainRow = find(rows, "GeneratorBench.openclChain", "10000000")
            val heavyRow = find(rows, "GeneratorBench.openclHeavy", "10000000")
            Seq(
              "OpenCL",
              fmt(chainRow),
              fmt(heavyRow),
              nx(rows, "GeneratorBench.spireChain", "10000000", chainRow.score),
              nx(rows, "GeneratorBench.spireHeavy", "10000000", heavyRow.score),
            )
          }, {
            val chainRow = find(rows, "GeneratorBench.plainChain", "10000000")
            val heavyRow = find(rows, "GeneratorBench.plainHeavy", "10000000")
            Seq(
              "plain while loop",
              fmt(chainRow),
              fmt(heavyRow),
              nx(rows, "GeneratorBench.spireChain", "10000000", chainRow.score),
              nx(rows, "GeneratorBench.spireHeavy", "10000000", heavyRow.score),
            )
          },
          Seq(
            "Spire on JVM",
            fmt(find(rows, "GeneratorBench.spireChain", "10000000")),
            fmt(find(rows, "GeneratorBench.spireHeavy", "10000000")),
            "1.0",
            "1.0",
          ), {
            val chainRow = find(rows, "GeneratorBench.breezeChain", "10000000")
            Seq(
              "Breeze",
              fmt(chainRow),
              "n/a",
              nx(rows, "GeneratorBench.spireChain", "10000000", chainRow.score),
              "n/a",
            )
          },
        ),
      ),
      "<!-- BENCHMARK_VERYHEAVY_TABLE -->" -> table(
        Seq("size", "OpenCL", "plain while loop", "Spire on JVM", "OpenCL vs Spire"),
        Seq("10000", "1000000", "10000000").map { size =>
          val openclRow = find(rows, "GeneratorBench.openclVeryHeavy", size)
          Seq(
            size,
            fmt(openclRow),
            fmt(find(rows, "GeneratorBench.plainVeryHeavy", size)),
            fmt(find(rows, "GeneratorBench.spireVeryHeavy", size)),
            nx(rows, "GeneratorBench.spireVeryHeavy", size, openclRow.score),
          )
        },
      ),
      "<!-- BENCHMARK_DEPTHSWEEP_TABLE -->" -> table(
        Seq("depth", "OpenCL", "plain", "Spire", "OpenCL vs Spire"),
        Seq("1", "8", "32", "128").map { depth =>
          val openclRow = find(rows, "DepthSweepBench.opencl", depth)
          Seq(
            depth,
            fmt(openclRow),
            fmt(find(rows, "DepthSweepBench.plain", depth)),
            fmt(find(rows, "DepthSweepBench.spire", depth)),
            nx(rows, "DepthSweepBench.spire", depth, openclRow.score),
          )
        },
      ),
      "<!-- BENCHMARK_PHASES_TABLE -->" -> table(
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
    )

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

  /** Row plus a `vs Spire` column, picking faster/slower automatically from which side of the Spire score `benchmark`/`param` lands on. */
  private def entryNx(
    rows: Seq[Row],
    benchmark: String,
    param: String,
    label: String,
    spireBenchmark: String,
  ): Seq[String] =
    val row = find(rows, benchmark, param)
    Seq(label, fmt(row), nx(rows, spireBenchmark, param, row.score))

  private def fmt(row: Row): String =
    s"${round(row.score)} +/- ${round(row.error)}"

  private def round(value: Double): String =
    if value >= 100 then f"$value%.0f" else if value >= 10 then f"$value%.1f" else f"$value%.2f"

  private def slowerThan(rows: Seq[Row], baseBenchmark: String, param: String, score: Double): String =
    f"${score / find(rows, baseBenchmark, param).score}%.1fx slower"

  private def fasterThan(rows: Seq[Row], baseBenchmark: String, param: String, score: Double): String =
    f"${find(rows, baseBenchmark, param).score / score}%.1fx faster"

  private def nx(rows: Seq[Row], spireBenchmark: String, param: String, score: Double): String =
    val spireScore = find(rows, spireBenchmark, param).score
    if score <= spireScore then fasterThan(rows, spireBenchmark, param, score)
    else slowerThan(rows, spireBenchmark, param, score)
