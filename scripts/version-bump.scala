#!/usr/bin/env scala-cli

//> using scala 3.3.4
//> using dep com.lihaoyi::os-lib:0.11.8

import os._

object VersionBump:

  def main(args: Array[String]): Unit =
    if args.isEmpty then
      println("Error: bump type required (major, minor, patch)")
      sys.exit(1)

    val bumpType = args.head.toLowerCase
    if !Seq("major", "minor", "patch").contains(bumpType) then
      println("Error: invalid bump type. Must be: major, minor, patch")
      sys.exit(1)

    val repoRoot = os.Path(os.proc("git", "rev-parse", "--show-toplevel").call().out.text().trim)
    val buildSbt = repoRoot / "build.sbt"
    val buildSc = repoRoot / "build.sc"
    val projectScala = repoRoot / "project.scala"

    var currentVersionOpt: Option[String] = None
    var targetFileOpt: Option[os.Path] = None
    var content = ""

    if os.exists(buildSbt) then
      targetFileOpt = Some(buildSbt)
      content       = os.read(buildSbt)
      val regex = "(?i)version\\s*:=\\s*\"(.*?)\"".r
      regex.findFirstMatchIn(content).foreach { m =>
        currentVersionOpt = Some(m.group(1))
      }
    else if os.exists(buildSc) then
      targetFileOpt = Some(buildSc)
      content       = os.read(buildSc)
      val regex = "(?i)def\\s*publishVersion\\s*=\\s*\"(.*?)\"".r
      val regex2 = "(?i)val\\s*version\\s*=\\s*\"(.*?)\"".r
      regex.findFirstMatchIn(content).orElse(regex2.findFirstMatchIn(content)).foreach { m =>
        currentVersionOpt = Some(m.group(1))
      }
    else if os.exists(projectScala) then
      targetFileOpt = Some(projectScala)
      content       = os.read(projectScala)
      val regex = "(?i)//\\s*version\\s*:=\\s*\"(.*?)\"".r
      regex.findFirstMatchIn(content).foreach { m =>
        currentVersionOpt = Some(m.group(1))
      }

    val (currentVersion, targetFile) = (currentVersionOpt, targetFileOpt) match
      case (Some(v), Some(f)) => (v, f)
      case _                  =>
        val defaultVer = "0.1.0"
        if os.exists(projectScala) then
          val updated = s"// version := \"$defaultVer\"\n" + os.read(projectScala)
          os.write.over(projectScala, updated)
          content = updated
          (defaultVer, projectScala)
        else if os.exists(buildSbt) then
          val updated = s"version := \"$defaultVer\"\n" + os.read(buildSbt)
          os.write.over(buildSbt, updated)
          content = updated
          (defaultVer, buildSbt)
        else
          println("Error: Could not locate build.sbt, build.sc, or project.scala to find version")
          sys.exit(1)

    println(s"Current version: $currentVersion")

    val parts = currentVersion.split('.').flatMap(_.toIntOption)
    if parts.length < 3 then
      println(s"Error: Version '$currentVersion' is not in standard semantic versioning format (X.Y.Z)")
      sys.exit(1)

    val Array(major, minor, patch) = parts.take(3)
    val nextVersion = bumpType match
      case "major" => s"${major + 1}.0.0"
      case "minor" => s"$major.${minor + 1}.0"
      case "patch" => s"$major.$minor.${patch + 1}"

    println(s"Bumping version to: $nextVersion")

    val updatedContent = targetFile match
      case f if f == buildSbt =>
        content.replaceFirst("version\\s*:=\\s*\".*?\"", s"version := \"$nextVersion\"")
      case f if f == buildSc =>
        if content.contains("def publishVersion") then
          content.replaceFirst(
            "def\\s*publishVersion\\s*=\\s*\".*?\"",
            s"def publishVersion = \"$nextVersion\"",
          )
        else content.replaceFirst("val\\s*version\\s*=\\s*\".*?\"", s"val version = \"$nextVersion\"")
      case f if f == projectScala =>
        content.replaceFirst("//\\s*version\\s*:=\\s*\".*?\"", s"// version := \"$nextVersion\"")
      case _ => content

    os.write.over(targetFile, updatedContent)
    println(s"✓ Updated version in ${targetFile.relativeTo(repoRoot)}")
