package docs

import java.nio.file.Files
import java.nio.file.Path
import java.nio.charset.StandardCharsets
import upickle.default.*

object BenchmarkResults:

  private final case class Row(benchmark: String, params: Map[String, String], score: Double, error: Double)

  def main(args: Array[String]): Unit =
    val jsonPath = Path.of(args(0))
    val markdownPath = Path.of(args(1))
    val rows = read[ujson.Value](Files.readString(jsonPath)).arr.toSeq.map { value =>
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

    val markdown = Files.readString(markdownPath)
    val generated = render(jsonPath.getFileName.toString, rows)
    Files.writeString(
      markdownPath,
      markdown.replace("<!-- BENCHMARK_RESULTS -->", generated),
      StandardCharsets.UTF_8,
    )

  private def render(sourceName: String, rows: Seq[Row]): String =
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
    ).mkString("\n")

  private def table(header: Seq[String], rows: Seq[Seq[String]]): String =
    (Seq(header, header.map(_ => "---")) ++ rows).map(_.mkString("| ", " | ", " |")).mkString("\n")

  private def matchRow(benchmark: String, param: String)(row: Row): Boolean =
    row.benchmark == benchmark && row.params.values.exists(_ == param)

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
