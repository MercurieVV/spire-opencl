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

  /** Per-launch host-side phase split, from the `PhaseCounters` aux counters `ClKernel.launch` emits — nanosecond totals over an iteration plus the
    * launch count to divide by. Absent for benchmarks that don't wire up `PhaseCounters` (the JVM rows, `openclPacked`/`openclPacking` share the
    * upload phase with the interleave). `readbackNs` is not "processing": almost all device time is absorbed there because the upload and NDRange
    * enqueue are non-blocking, so it is reported alongside upload and launch rather than folded into either.
    */
  private final case class Phases(uploadNs: Double, launchNs: Double, readbackNs: Double, ops: Double):
    def uploadPerOp: Double = uploadNs / ops
    def launchPerOp: Double = launchNs / ops
    def readbackPerOp: Double = readbackNs / ops

  private final case class Row(
    name: String,
    params: Map[String, String],
    score: Double,
    unit: String,
    elements: Long,
    profile: Traffic.Profile,
    phases: Option[Phases]):

    /** Every score here is average time per operation; JMH reports it in whatever unit the benchmark asked for. */
    private def seconds: Double = unit match
      case u if u.startsWith("ns") => score * 1e-9
      case u if u.startsWith("us") => score * 1e-6
      case u if u.startsWith("ms") => score * 1e-3
      case _                       => score

    def elementsPerSecond: Double = elements / seconds
    def deviceGBs: Double = elements.toDouble * profile.deviceBytes / seconds / 1e9
    def hostUpGBs: Double = elements.toDouble * profile.hostUp / seconds / 1e9
    def hostDownGBs: Double = elements.toDouble * profile.hostDown / seconds / 1e9
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
        val secondary = r.obj.get("secondaryMetrics").map(_.obj)
        val phases = secondary.flatMap { sm =>
          for
            upload   <- sm.get("uploadNs")
            launch   <- sm.get("launchNs")
            readback <- sm.get("readbackNs")
            ops      <- sm.get("ops")
          yield Phases(
            upload("score").num,
            launch("score").num,
            readback("score").num,
            ops("score").num,
          )
        }
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
          phases,
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
      println("  crossing the API boundary per launch, upload and download kept apart (0 MB means that leg does not happen, not that it is free):")
      val tw = transfers.map(_.label.length).max.min(62)
      println("  " + "benchmark".padTo(tw, ' ') + "        upload MB  GB/s      download MB  GB/s")
      transfers.sortBy(r => (r.name, r.elements)).foreach { r =>
        val upMB = r.elements * r.profile.hostUp / 1e6
        val downMB = r.elements * r.profile.hostDown / 1e6
        println(f"  ${r.label.padTo(tw, ' ')} ${upMB}%9.1f MB ${r.hostUpGBs}%6.1f GB/s   ${downMB}%9.1f MB ${r.hostDownGBs}%6.1f GB/s")
      }

    val phased = rows.filter(_.phases.isDefined)
    if phased.nonEmpty then
      println()
      println("  host-side phase split per launch (upload = params to device, readback dominated by the blocking wait, not device time):")
      val pw = phased.map(_.label.length).max.min(62)
      println("  " + "benchmark".padTo(pw, ' ') + "     upload    launch  readback")
      phased.sortBy(r => (r.name, r.elements)).foreach { r =>
        val p = r.phases.get
        println(f"  ${r.label.padTo(pw, ' ')} ${p.uploadPerOp / 1e3}%8.1fus ${p.launchPerOp / 1e3}%7.1fus ${p.readbackPerOp / 1e3}%8.1fus")
      }
