#!/usr/bin/env scala-cli

//> using scala 3.3.4
//> using dep com.lihaoyi::os-lib:0.11.8

import os._

object GitPrePush:

  def main(args: Array[String]): Unit =
    val repoRoot = os.Path(os.proc("git", "rev-parse", "--show-toplevel").call().out.text().trim)

    val buildTool = if os.exists(repoRoot / "build.sbt") then "sbt"
    else if os.exists(repoRoot / "build.sc") || os.exists(repoRoot / "build.mill") then "mill"
    else "scala-cli"

    println("=== Git Pre-Push Verification Checks ===")
    println(s"Running compilation and tests using $buildTool...")

    val exitCode = buildTool match
      case "sbt" =>
        val hasPrePushAlias = os.read(repoRoot / "build.sbt").contains("addCommandAlias(\"prePush\"")
        val cmd = if hasPrePushAlias then Seq("sbt", "prePush") else Seq("sbt", "compile", "test")
        os.proc(cmd).call(cwd = repoRoot, check = false).exitCode

      case "mill" =>
        val buildFile = if os.exists(repoRoot / "build.mill") then repoRoot / "build.mill" else repoRoot / "build.sc"
        val buildContent = os.read(buildFile)
        val cmd = if buildContent.contains("def prePush") then Seq("mill", "prePush")
        else if buildContent.contains("object scalafix") then Seq("mill", "scalafix.test")
        else Seq("mill", "app.test")
        os.proc(cmd).call(cwd = repoRoot, check = false).exitCode

      case "scala-cli" =>
        os.proc("scala-cli", "test", ".").call(cwd = repoRoot, check = false).exitCode

    if exitCode != 0 then
      println("\n[ERROR] Pre-push verification failed! Push aborted.")
      sys.exit(1)

    val stainlessScript = repoRoot / "scripts" / "stainless-verify.sh"
    if os.exists(stainlessScript) then
      println("Running Stainless formal verification...")
      val stainlessExit = os.proc("bash", stainlessScript.toString).call(cwd = repoRoot, check = false).exitCode
      if stainlessExit != 0 then
        println("\n[ERROR] Stainless verification failed! Push aborted.")
        sys.exit(1)

    println("✓ All pre-push checks passed successfully!")
