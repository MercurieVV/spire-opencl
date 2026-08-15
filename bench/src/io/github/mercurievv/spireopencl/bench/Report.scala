package io.github.mercurievv.spireopencl.bench

/** Turns a recorded JMH report into rates.
  *
  * A duration answers "how long", which is only useful against a decision. A rate answers "how close to the limit", which is what says whether there
  * is anything left to win — and the run already contains everything needed to compute one, since `Traffic` knows what each benchmark moves.
  *
  * The ceiling is the best bandwidth **observed in this run**, not a vendor figure. A device's peak is not queryable through OpenCL, and quoting a
  * marketing number would make every row look worse than the hardware really allows here. The best row actually achieved is a floor on what is
  * achievable, which is the honest comparison and moves with the machine the run happened on.
  *
  * Usage: `Report <report.json>`.
  */
object Report:

  /** Past any cache on a machine this benchmark suite is meant to run on, so a row above it is answering out of main memory. */
  private val CacheProofBytes = 64L << 20

  private final case class Row(
    name: String,
    params: Map[String, String],
    score: Double,
    unit: String,
    elements: Long,
    profile: Traffic.Profile):

    /** Every score here is average time per operation; JMH reports it in whatever unit the benchmark asked for. */
    private def seconds: Double = unit match
      case u if u.startsWith("ns") => score * 1e-9
      case u if u.startsWith("us") => score * 1e-6
      case u if u.startsWith("ms") => score * 1e-3
      case _                       => score

    def elementsPerSecond: Double = elements / seconds
    def deviceGBs: Double = elements.toDouble * profile.deviceBytes / seconds / 1e9
    def hostGBs: Double = elements.toDouble * profile.hostBytes / seconds / 1e9
    def gops: Double = elements.toDouble * profile.ops / seconds / 1e9
    def label: String = if params.isEmpty then name else s"$name (${params.toList.sortBy(_._1).map((k, v) => s"$k=$v").mkString(", ")})"

  private def read(path: os.Path): List[Row] =
    ujson
      .read(os.read(path))
      .arr
      .toList
      .flatMap { r =>
        val name = r("benchmark").str.split('.').takeRight(2).mkString(".")
        val params = r.obj.get("params").map(_.obj.view.mapValues(_.str).toMap).getOrElse(Map.empty[String, String])
        for
          profile <- Traffic.of(name, params)
          n       <- Traffic.elements(name, params)
        yield Row(
          name,
          params,
          r("primaryMetric")("score").num,
          r("primaryMetric")("scoreUnit").str,
          n,
          profile,
        )
      }

  def main(args: Array[String]): Unit =
    if args.isEmpty then
      System.err.println("usage: Report <report.json>")
      sys.exit(2)

    val rows = read(os.Path(args(0), os.pwd))
    if rows.isEmpty then
      System.err.println("no benchmarks in the report have a known traffic shape; see Traffic.of")
      sys.exit(2)

    /* The ceiling comes only from rows whose working set is far past any cache. A 10^4-element row touches
     * 160 kB, answers out of L2, and reaches a bandwidth main memory cannot sustain -- taking it as the
     * ceiling would make every honest row look like a fraction of a number that was never available to
     * them. Among the rows that do go to memory, the maximum is the ceiling: a compute-bound row moves few
     * bytes slowly and must not drag it down. */
    val cacheProof = rows.filter(r => r.elements * r.profile.deviceBytes > CacheProofBytes)
    val ceiling = cacheProof.map(_.deviceGBs).filter(d => !d.isNaN && d.isFinite).maxOption.getOrElse(Double.NaN)

    println(s"ceiling: ${f"$ceiling%.1f"} GB/s, the best achieved by any row with a working set past ${CacheProofBytes / (1 << 20)} MB")
    println("  rows below that size are cache-resident and can exceed it; their %ceil is not a shortfall")
    println("ops counts a transcendental as one, so the compute-bound rows are a lower bound")
    println()

    val width = rows.map(_.label.length).max.min(62)
    println(
      "  " + "benchmark".padTo(width, ' ') + "      elem/s      GB/s   %ceil     Gop/s   op/byte",
    )

    rows.sortBy(r => (r.name, r.elements)).foreach { r =>
      val pct = if ceiling.isNaN || ceiling == 0 then Double.NaN else r.deviceGBs / ceiling * 100.0
      println(
        f"  ${r.label.padTo(width, ' ')} ${r.elementsPerSecond / 1e9}%8.2fG ${r.deviceGBs}%9.1f ${pct}%6.1f%% ${r.gops}%9.2f ${r.profile.intensity}%9.2f",
      )
    }

    val transfers = rows.filter(_.profile.hostBytes > 0)
    if transfers.nonEmpty then
      println()
      println("  crossing the API boundary per launch:")
      val tw = transfers.map(_.label.length).max.min(62)
      transfers.sortBy(r => (r.name, r.elements)).foreach { r =>
        println(f"  ${r.label.padTo(tw, ' ')} ${r.elements * r.profile.hostBytes / 1e6}%9.1f MB ${r.hostGBs}%8.1f GB/s")
      }
