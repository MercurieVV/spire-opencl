package io.github.mercurievv.spireopencl.bench

/** Compares two recorded JMH runs and fails on regression.
  *
  * Neither JMH nor Mill has a history story — JMH writes a JSON report and forgets it, and there is no built-in notion of a previous run to compare
  * against. The usual answer, `benchmark-action/github-action-benchmark`, keeps history on a CI branch, which is no use here: the runners have no GPU
  * and pocl numbers would describe the runner. So history is local, one JSON per run under `bench/results`, and this compares two of them.
  *
  * A regression is **not** just a slower number. Benchmarks are noisy, and a threshold on its own turns every noisy row into a false alarm. A row is
  * reported as regressed only when it is worse by more than the threshold *and* the two runs' confidence intervals do not overlap — so the claim is
  * that the difference is larger than the run-to-run spread each run measured for itself, not merely larger than an arbitrary percentage.
  *
  * Usage: `History <baseline.json> <current.json> [thresholdPercent]`. Exits 1 if anything regressed, so it can gate a commit.
  */
object History:

  private val DefaultThreshold = 10.0

  /** One row of a report, keyed the way JMH identifies a measurement: the benchmark plus the parameter values it ran with. */
  private final case class Row(
    name: String,
    params: String,
    score: Double,
    lo: Double,
    hi: Double,
    unit: String,
    lowerIsBetter: Boolean):
    def key: String = if params.isEmpty then name else s"$name ($params)"

  /** JMH writes `scoreError` as the string `"NaN"` when it has too little data to compute one — every aux counter, and any single-iteration run. A
    * plain numeric read would throw on exactly the rows a comparison most needs to treat gently.
    */
  private def num(v: ujson.Value): Double = v match
    case ujson.Num(d) => d
    case ujson.Str(s) => s.toDoubleOption.getOrElse(Double.NaN)
    case _            => Double.NaN

  private def parse(path: os.Path): Map[String, Row] =
    ujson
      .read(os.read(path))
      .arr
      .map { r =>
        val metric = r("primaryMetric")
        val mode = r("mode").str
        // `thrpt` counts operations per unit time, so a bigger number is better; every other mode measures time.
        val lowerIsBetter = mode != "thrpt"
        val score = num(metric("score"))
        val ci = metric.obj.get("scoreConfidence").map(_.arr.map(num)).filter(_.length == 2)
        val (lo, hi) = ci.map(c => (c(0), c(1))).getOrElse((Double.NaN, Double.NaN))
        val params = r.obj
          .get("params")
          .map(_.obj.toList.sortBy(_._1).map((k, v) => s"$k=${v.str}").mkString(", "))
          .getOrElse("")
        val row = Row(
          r("benchmark").str.split('.').takeRight(2).mkString("."),
          params,
          score,
          lo,
          hi,
          metric("scoreUnit").str,
          lowerIsBetter,
        )
        row.key -> row
      }
      .toMap

  /** Disjoint confidence intervals. `NaN` bounds — a run with too few iterations to have any — count as overlapping, so an unmeasurable difference is
    * never reported as a real one.
    */
  private def separated(a: Row, b: Row): Boolean =
    if a.lo.isNaN || a.hi.isNaN || b.lo.isNaN || b.hi.isNaN then false
    else a.hi < b.lo || b.hi < a.lo

  private def deltaPercent(baseline: Row, current: Row): Double =
    val raw = (current.score - baseline.score) / baseline.score * 100.0
    // Reported so that negative always means "better", whichever direction the mode counts in.
    if baseline.lowerIsBetter then raw else -raw

  def main(args: Array[String]): Unit =
    if args.length < 2 then
      System.err.println("usage: History <baseline.json> <current.json> [thresholdPercent]")
      sys.exit(2)

    val baselinePath = os.Path(args(0), os.pwd)
    val currentPath = os.Path(args(1), os.pwd)
    val threshold = args.lift(2).flatMap(_.toDoubleOption).getOrElse(DefaultThreshold)

    val baseline = parse(baselinePath)
    val current = parse(currentPath)

    println(s"baseline  ${baselinePath.last}")
    println(s"current   ${currentPath.last}")
    println(s"threshold ${threshold}%, and the confidence intervals must not overlap")
    println()

    val shared = baseline.keySet.intersect(current.keySet).toList.sorted
    val added = current.keySet.diff(baseline.keySet).toList.sorted
    val removed = baseline.keySet.diff(current.keySet).toList.sorted

    val width = (shared ++ added ++ removed).map(_.length).maxOption.getOrElse(20).min(70)

    val judged = shared.map { k =>
      val (b, c) = (baseline(k), current(k))
      val d = deltaPercent(b, c)
      val verdict =
        if d > threshold && separated(b, c) then "REGRESSED"
        else if d < -threshold && separated(b, c) then "improved"
        else "="
      (k, b, c, d, verdict)
    }

    judged.foreach {
      (
        k,
        b,
        c,
        d,
        verdict,
      ) =>
        val mark = if verdict == "REGRESSED" then " <<<" else ""
        println(f"  ${k.padTo(width, ' ')}  ${b.score}%12.1f -> ${c.score}%12.1f ${b.unit}%-8s ${d}%+7.1f%%$mark")
    }

    val blank = " " * 12
    added.foreach(k => println(f"  ${k.padTo(width, ' ')}  $blank -> ${current(k).score}%12.1f ${current(k).unit}%-8s      new"))

    /* Listed as a count, not row by row. Running a subset — `bench.record GeneratorBench` — is the normal
     * way to use this, and every benchmark the filter excluded would otherwise be printed as if it had
     * been deleted, burying the rows that were actually measured. */
    if removed.nonEmpty then
      println()
      println(s"${removed.length} in the baseline were not measured here (a filtered run explains this; `${removed.head}` was the first)")

    val regressed = judged.filter(_._5 == "REGRESSED")
    val improved = judged.filter(_._5 == "improved")

    println()
    println(
      s"${judged.length - regressed.length - improved.length} unchanged, ${improved.length} improved, " +
        s"${regressed.length} regressed, ${added.length} new, ${removed.length} not measured",
    )

    if regressed.nonEmpty then
      println()
      println(s"REGRESSED (${regressed.length}):")
      regressed.foreach {
        (
          k,
          b,
          c,
          d,
          _,
        ) =>
          println(f"  $k: ${b.score}%.1f -> ${c.score}%.1f ${b.unit} ($d%+.1f%%, ${b.lo}%.1f..${b.hi}%.1f vs ${c.lo}%.1f..${c.hi}%.1f)")
      }
      sys.exit(1)
    else println("no regressions")
