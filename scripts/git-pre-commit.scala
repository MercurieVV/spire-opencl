#!/usr/bin/env scala-cli

//> using scala 3.3.4
//> using dep com.lihaoyi::os-lib:0.11.8

import os._

object GitPreCommit:

  def main(args: Array[String]): Unit =
    val repoRoot = os.Path(os.proc("git", "rev-parse", "--show-toplevel").call().out.text().trim)

    val buildTool = if os.exists(repoRoot / "build.sbt") then "sbt"
    else if os.exists(repoRoot / "build.sc") || os.exists(repoRoot / "build.mill") then "mill"
    else "scala-cli"

    val hasScalafmt = os.exists(repoRoot / ".scalafmt.conf")
    val hasScalafix = os.exists(repoRoot / ".scalafix.conf")

    if !hasScalafmt && !hasScalafix then
      println("No formatting or linting configuration found. Pre-commit check skipped.")
      sys.exit(0)

    println("=== Git Pre-Commit Quality Checks ===")

    if hasScalafmt then
      println("Checking code formatting (Scalafmt)...")
      val fmtExit = buildTool match
        case "sbt" =>
          os.proc("sbt", "scalafmtCheckAll").call(cwd = repoRoot, check = false).exitCode
        case "scala-cli" =>
          os.proc("scala-cli", "fmt", "--check", ".").call(cwd = repoRoot, check = false).exitCode
        case "mill" =>
          os.proc("mill", "mill.scalalib.scalafmt.ScalafmtModule/checkFormatAll").call(cwd = repoRoot, check = false).exitCode

      if fmtExit != 0 then
        println("\n[ERROR] Code formatting check failed!")
        println("Please run the formatting tool to fix it:")
        buildTool match
          case "sbt"       => println("  sbt scalafmtAll")
          case "scala-cli" => println("  scala-cli fmt .")
          case "mill"      => println("  mill mill.scalalib.scalafmt.ScalafmtModule/reformatAll")
        sys.exit(1)

    if hasScalafix then
      println("Checking code linting (Scalafix)...")
      val lintExit = buildTool match
        case "sbt" =>
          os.proc("sbt", "scalafixAll --check").call(cwd = repoRoot, check = false).exitCode
        case "scala-cli" =>
          val targets = Seq("app/src", "app/test/src").filter(p => os.exists(repoRoot / os.RelPath(p)))
          if targets.nonEmpty then
            // `scala-cli --power fix` must be given `.` (not explicit dirs) so it
            // picks up root project.scala's dependencies, incl. test.dep. That
            // also makes it walk excluded scripts (Setup.scala, scripts/, ...)
            // that were never compiled, so filter out the resulting noise.
            val excludePrefixes =
              val projectScalaPath = repoRoot / "project.scala"
              if os.exists(projectScalaPath) then
                val excludeRegex = "//>\\s*using\\s+exclude\\s+(.+)".r
                os.read
                  .lines(projectScalaPath)
                  .flatMap { line =>
                    excludeRegex.findFirstMatchIn(line.trim).map(_.group(1).trim)
                  }
                  .toList
              else Nil
            val result = os
              .proc("scala-cli", "--power", "fix", "--check", ".")
              .call(cwd = repoRoot, check = false, stdout = os.Pipe, stderr = os.Pipe, mergeErrIntoOut = true)
            if result.exitCode == 0 then 0
            else
              val noiseRegex = "^error: SemanticDB not found: (.+)$".r
              def isNoise(line: String): Boolean =
                noiseRegex.findFirstMatchIn(line) match
                  case Some(m) =>
                    val path = m.group(1)
                    excludePrefixes.exists(prefix => path == prefix || path.startsWith(prefix + "/"))
                  case None => false
              val meaningfulLines = result.out.text().linesIterator.toList.filterNot(isNoise)
              val hasRealError = meaningfulLines.exists(l => l.contains("error:") || l.trim.startsWith("[error]") || l.startsWith("--- "))
              if hasRealError then
                meaningfulLines.foreach(println)
                1
              else
                println("✓ Linting clean (ignored known noise from excluded scripts)")
                0
          else 0
        case "mill" =>
          // `__.fix` is mill-scalafix's own task (com.goyeau.mill.scalafix.ScalafixModule), reached through the
          // `__` wildcard selector so this hook needs no per-repo module name. A repo with no ScalafixModule
          // mixed into any module makes `fix` unresolvable across the board, which is Mill's ordinary "nothing
          // here" outcome, not a broken build.
          val res = os
            .proc("mill", "__.fix", "--check")
            .call(cwd = repoRoot, check = false, stdout = os.Pipe, stderr = os.Pipe)
          if res.exitCode != 0 then
            val errText = res.err.text()
            val outText = res.out.text()
            val cleanErr = errText.replaceAll("\u001b\\[[;\\d]*m", "")
            val cleanOut = outText.replaceAll("\u001b\\[[;\\d]*m", "")
            if cleanErr.contains("Cannot resolve") || cleanOut.contains("Cannot resolve") then
              println("✓ Scalafix is not configured in Mill. Skipping Scalafix check.")
              0
            else
              System.err.print(errText)
              System.out.print(outText)
              res.exitCode
          else 0

      if lintExit != 0 then
        println("\n[ERROR] Code linting check failed!")
        println("Please run linting to fix it:")
        buildTool match
          case "sbt"       => println("  sbt scalafixAll")
          case "scala-cli" => println("  scala-cli --power fix .")
          case "mill"      => println("  mill mill.scalalib.contrib.ScalafixModule/fix")
        sys.exit(1)

    println("✓ All pre-commit checks passed successfully!")
