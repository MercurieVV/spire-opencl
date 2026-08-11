#!/usr/bin/env scala-cli

//> using scala 3.3.4
//> using dep com.lihaoyi::os-lib:0.11.8

import os._

object DependencyUpdate:

  def main(args: Array[String]): Unit =
    var applyUpdates = false
    var targetPathOpt: Option[os.Path] = None
    var showHelp = false

    args.foreach {
      case "--apply" | "-a"           => applyUpdates = true
      case "--help" | "-h"            => showHelp     = true
      case arg if arg.startsWith("-") =>
        System.err.println(s"Error: Unknown option $arg")
        showHelp = true
      case arg =>
        if targetPathOpt.isDefined then
          System.err.println(s"Error: Multiple target directories specified.")
          showHelp = true
        else
          try targetPathOpt = Some(os.Path(arg, os.pwd))
          catch {
            case _: Exception =>
              System.err.println(s"Error: Invalid path '$arg'")
              sys.exit(1)
          }
    }

    val usage =
      """Check and apply dependency updates in a target project without Git.
        |
        |Usage:
        |  scala-cli run scripts/dependency-update.scala -- [options] [target-directory]
        |
        |Options:
        |  --apply, -a   Apply updates to files if supported by the build tool.
        |  --help, -h    Show this help message.
        |
        |Supported build tools:
        |  - Mill (build.sc or build.mill)
        |  - Scala CLI (project.scala)
        |  - SBT (build.sbt)
        |  - Maven (pom.xml)
        |  - Gradle (build.gradle or build.gradle.kts)
        |""".stripMargin

    if showHelp then
      println(usage)
      sys.exit(0)

    val targetDir = targetPathOpt.getOrElse(os.pwd)
    if !os.exists(targetDir) || !os.isDir(targetDir) then
      System.err.println(
        s"Error: Target directory '$targetDir' does not exist or is not a directory.",
      )
      sys.exit(1)

    println(s"Analyzing target project at $targetDir...")

    // Detect build tools
    val isMill =
      os.exists(targetDir / "build.sc") || os.exists(targetDir / "build.mill")
    val isSbt = os.exists(targetDir / "build.sbt")
    val isScalaCli = os.exists(targetDir / "project.scala")
    val isMaven = os.exists(targetDir / "pom.xml")
    val isGradle = os.exists(targetDir / "build.gradle") || os.exists(
      targetDir / "build.gradle.kts",
    )

    val buildTool =
      if isMill then Some("mill")
      else if isSbt then Some("sbt")
      else if isScalaCli then Some("scala-cli")
      else if isMaven then Some("maven")
      else if isGradle then Some("gradle")
      else None

    buildTool match {
      case None =>
        System.err.println(
          "Error: Could not detect any supported build tool (Mill, SBT, Scala CLI, Maven, Gradle) in the target directory.",
        )
        sys.exit(1)
      case Some(tool) =>
        println(s"Detected build tool: $tool")
        runForTool(tool, targetDir, applyUpdates)
    }

  private def runForTool(
    tool: String,
    targetDir: os.Path,
    applyUpdates: Boolean,
  ): Unit =
    tool match {
      case "mill" =>
        val cmd = if os.exists(targetDir / "mill") then "./mill" else "mill"
        if applyUpdates then
          println(
            "Warning: Mill does not support automatic in-place dependency updates.",
          )
          println(
            "Running update check. You will need to manually apply the updates to build.mill / build.sc.",
          )
        println(s"Running: $cmd mill.javalib.Dependency/showUpdates")
        val res = os
          .proc(cmd, "mill.javalib.Dependency/showUpdates")
          .call(
            cwd    = targetDir,
            check  = false,
            stdout = os.Inherit,
            stderr = os.Inherit,
          )
        sys.exit(res.exitCode)

      case "scala-cli" =>
        val baseCmd = Seq("scala-cli", "--power", "dependency-update")
        val cmd = if applyUpdates then baseCmd :+ "--all" else baseCmd
        println(s"Running: ${cmd.mkString(" ")} .")
        val res = os
          .proc(cmd :+ ".")
          .call(
            cwd    = targetDir,
            check  = false,
            stdout = os.Inherit,
            stderr = os.Inherit,
          )
        sys.exit(res.exitCode)

      case "sbt" =>
        if applyUpdates then
          println("Running update using sbt-dependency-updater plugin...")
          println("Running: sbt updateDependencies")
          val res = os
            .proc("sbt", "updateDependencies")
            .call(
              cwd    = targetDir,
              check  = false,
              stdout = os.Inherit,
              stderr = os.Inherit,
            )
          if res.exitCode != 0 then
            println(
              "Note: 'sbt updateDependencies' failed. Make sure you have the 'sbt-dependency-updater' plugin installed.",
            )
          sys.exit(res.exitCode)
        else
          println("Checking updates using sbt-updates plugin...")
          println("Running: sbt dependencyUpdates")
          val res = os
            .proc("sbt", "dependencyUpdates")
            .call(
              cwd    = targetDir,
              check  = false,
              stdout = os.Inherit,
              stderr = os.Inherit,
            )
          if res.exitCode != 0 then
            println(
              "Note: 'sbt dependencyUpdates' failed. Make sure you have the 'sbt-updates' plugin installed.",
            )
          sys.exit(res.exitCode)

      case "maven" =>
        val cmd = if applyUpdates then Seq("mvn", "versions:use-latest-releases")
        else Seq("mvn", "versions:display-dependency-updates")
        println(s"Running: ${cmd.mkString(" ")}")
        val res = os
          .proc(cmd)
          .call(
            cwd    = targetDir,
            check  = false,
            stdout = os.Inherit,
            stderr = os.Inherit,
          )
        sys.exit(res.exitCode)

      case "gradle" =>
        val gradlew =
          if os.exists(targetDir / "gradlew") then "./gradlew" else "gradle"
        if applyUpdates then
          val hasCatalog =
            os.exists(targetDir / "gradle" / "libs.versions.toml")
          if hasCatalog then
            println(s"Running: $gradlew versionCatalogUpdate")
            val res = os
              .proc(gradlew, "versionCatalogUpdate")
              .call(
                cwd    = targetDir,
                check  = false,
                stdout = os.Inherit,
                stderr = os.Inherit,
              )
            if res.exitCode != 0 then
              println(
                "Note: 'versionCatalogUpdate' failed. Make sure the 'littlerobots.version-catalog-update' plugin is configured in your project.",
              )
            sys.exit(res.exitCode)
          else
            println(
              "Error: Gradle does not support automatic in-place updates for build.gradle files.",
            )
            println(
              "Please move your dependencies to a version catalog (gradle/libs.versions.toml) and use version-catalog-update-plugin.",
            )
            sys.exit(1)
        else
          println(s"Running: $gradlew dependencyUpdates")
          val res = os
            .proc(gradlew, "dependencyUpdates")
            .call(
              cwd    = targetDir,
              check  = false,
              stdout = os.Inherit,
              stderr = os.Inherit,
            )
          if res.exitCode != 0 then
            println(
              "Note: 'dependencyUpdates' failed. Make sure you have the 'com.github.ben-manes.versions' plugin configured in your Gradle project.",
            )
          sys.exit(res.exitCode)
    }
