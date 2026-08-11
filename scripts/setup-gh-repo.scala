#!/usr/bin/env scala-cli

//> using scala 3.8.4
//> using options -Ysemanticdb
//> using options -sourceroot:.
//> using dep com.lihaoyi::os-lib:0.11.8

import os._

object SetupGhRepo:

  private val releaseSecrets =
    List(
      "SONATYPE_USERNAME",
      "SONATYPE_PASSWORD",
      "PGP_SECRET_KEY",
      "PGP_SECRET_KEY_PASSWORD",
    )

  final case class Config(
    repo: Option[String],
    opVault: Option[String],
    opItem: Option[String],
    opItemName: Option[String],
    opGpgItem: Option[String],
    opGpgItemName: Option[String],
    opSonatypeItem: Option[String],
    opSonatypeItemName: Option[String],
    opFallbackItemNames: List[String],
    verbose: Boolean)

  def main(args: Array[String]): Unit =
    parse(args.toList) match
      case Left(message) =>
        Console.err.println(message)
        Console.err.println("")
        printUsage()
        sys.exit(1)

      case Right(config) if args.contains("--help") || args.contains("-h") =>
        printUsage()

      case Right(config) =>
        requireCommand("gh")
        requireCommand("git")

        val repo = config.repo.orElse(inferRepo()).getOrElse {
          Console.err.println(
            "REPO is empty: pass --repo OWNER/REPO or run from a GitHub-backed git checkout.",
          )
          sys.exit(1)
        }

        releaseSecrets.foreach { secret =>
          val value = resolveValue(secret, config).getOrElse {
            Console.err.println(missingValueMessage(secret))
            sys.exit(1)
          }
          os.proc(
            "gh",
            "secret",
            "set",
            secret,
            "--repo",
            repo,
            "--body",
            value,
          ).call(env = ghNoPagerEnv)
          println(s"Set repo secret $secret")
        }

        val vulnerabilityAlerts = os
          .proc(
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "--method",
            "PUT",
            s"/repos/$repo/vulnerability-alerts",
          )
          .call(
            check  = false,
            stdout = os.Pipe,
            stderr = os.Pipe,
            env    = ghNoPagerEnv,
          )

        if vulnerabilityAlerts.exitCode != 0 then
          Console.err.print(vulnerabilityAlerts.err.text())
          sys.exit(vulnerabilityAlerts.exitCode)

        println("Enabled vulnerability alerts and dependency graph")
        println(
          s"Done. Repo $repo is ready for tag-driven Maven Central releases.",
        )

  private def parse(args: List[String]): Either[String, Config] =
    def loop(rest: List[String], config: Config): Either[String, Config] =
      rest match
        case Nil                       => Right(config)
        case ("-h" | "--help") :: Nil  => Right(config)
        case "--repo" :: value :: tail =>
          loop(tail, config.copy(repo = nonEmpty(value)))
        case "--op-vault" :: value :: tail =>
          loop(tail, config.copy(opVault = nonEmpty(value)))
        case "--op-item" :: value :: tail =>
          loop(tail, config.copy(opItem = nonEmpty(value)))
        case "--op-item-name" :: value :: tail =>
          loop(tail, config.copy(opItemName = nonEmpty(value)))
        case "--op-gpg-item" :: value :: tail =>
          loop(tail, config.copy(opGpgItem = nonEmpty(value)))
        case "--op-gpg-item-name" :: value :: tail =>
          loop(tail, config.copy(opGpgItemName = nonEmpty(value)))
        case "--op-sonatype-item" :: value :: tail =>
          loop(tail, config.copy(opSonatypeItem = nonEmpty(value)))
        case "--op-sonatype-item-name" :: value :: tail =>
          loop(tail, config.copy(opSonatypeItemName = nonEmpty(value)))
        case "--op-fallback-item" :: value :: tail =>
          loop(
            tail,
            config.copy(opFallbackItemNames =
              config.opFallbackItemNames ++ nonEmpty(value).toList,
            ),
          )
        case "--verbose" :: tail         => loop(tail, config.copy(verbose = true))
        case "--repo" :: Nil             => Left("Missing value for --repo")
        case "--op-vault" :: Nil         => Left("Missing value for --op-vault")
        case "--op-item" :: Nil          => Left("Missing value for --op-item")
        case "--op-item-name" :: Nil     => Left("Missing value for --op-item-name")
        case "--op-gpg-item" :: Nil      => Left("Missing value for --op-gpg-item")
        case "--op-gpg-item-name" :: Nil =>
          Left("Missing value for --op-gpg-item-name")
        case "--op-sonatype-item" :: Nil =>
          Left("Missing value for --op-sonatype-item")
        case "--op-sonatype-item-name" :: Nil =>
          Left("Missing value for --op-sonatype-item-name")
        case "--op-fallback-item" :: Nil =>
          Left("Missing value for --op-fallback-item")
        case unknown :: _ => Left(s"Unknown argument: $unknown")

    loop(
      args,
      Config(
        repo                = env("REPO"),
        opVault             = env("OP_VAULT"),
        opItem              = env("OP_ITEM"),
        opItemName          = env("OP_ITEM_NAME"),
        opGpgItem           = env("OP_GPG_ITEM"),
        opGpgItemName       = env("OP_GPG_ITEM_NAME"),
        opSonatypeItem      = env("OP_SONATYPE_ITEM"),
        opSonatypeItemName  = env("OP_SONATYPE_ITEM_NAME"),
        opFallbackItemNames = envList("OP_ITEM_NAMES"),
        verbose             = false,
      ),
    )

  private def resolveValue(name: String, config: Config): Option[String] =
    env(name)
      .orElse(
        env(s"OP_${name}_REF").flatMap(ref => readOnePassword(ref, config)),
      )
      .orElse(
        itemRefs(name, config)
          .to(LazyList)
          .flatMap(ref => readOnePassword(ref, config))
          .headOption,
      )
      .orElse(
        defaultItemNames(name, config)
          .to(LazyList)
          .flatMap(item => readOnePasswordItemField(item, name, config))
          .headOption,
      )

  private def itemRefs(name: String, config: Config): List[String] =
    val explicit =
      itemBaseRefs(name, config).map(base => s"$base/$name")
    val named =
      itemNames(name, config).flatMap(itemName => config.opVault.map(vault => s"op://$vault/$itemName/$name"))
    explicit ++ named

  private def itemBaseRefs(name: String, config: Config): List[String] =
    val groupBase =
      if name.startsWith("PGP_") then config.opGpgItem.toList
      else if name.startsWith("SONATYPE_") then config.opSonatypeItem.toList
      else Nil
    groupBase ++ config.opItem.toList

  private def itemNames(name: String, config: Config): List[String] =
    val groupName =
      if name.startsWith("PGP_") then config.opGpgItemName.toList
      else if name.startsWith("SONATYPE_") then config.opSonatypeItemName.toList
      else Nil
    groupName ++ config.opItemName.toList

  private def defaultItemNames(name: String, config: Config): List[String] =
    val grouped =
      if name.startsWith("SONATYPE_") then
        List(
          "SONATYPE_CREDS",
          "scala-llm-template-sonatype",
          "sonatype",
          "maven-central",
        )
      else if name.startsWith("PGP_") then List("GPG", "scala-llm-template-gpg", "gpg", "pgp")
      else Nil
    val common =
      List(
        "scala-llm-template",
        "scala-llm-template-release",
        "scala-llm-template releases",
        "github-release",
        "github release",
      )
    (itemNames(
      name,
      config,
    ) ++ config.opFallbackItemNames ++ grouped ++ common).distinct

  private def readOnePassword(ref: String, config: Config): Option[String] =
    requireCommand("op")
    val result =
      os.proc("op", "read", ref).call(check = false, stderr = os.Pipe)
    if result.exitCode == 0 then nonEmpty(result.out.text().trim)
    else
      if config.verbose then
        Console.err.println(s"1Password read failed for $ref:")
        Console.err.print(result.err.text())
      None

  private def readOnePasswordItemField(
    item: String,
    name: String,
    config: Config,
  ): Option[String] =
    requireCommand("op")

    fieldNames(name)
      .to(LazyList)
      .flatMap { field =>
        val vaultArgs =
          config.opVault.toList.flatMap(vault => List("--vault", vault))
        val result =
          os.proc(
            List(
              "op",
              "item",
              "get",
              item,
              "--fields",
              s"label=$field",
              "--reveal",
            ) ++ vaultArgs,
          ).call(check = false, stderr = os.Pipe)

        if result.exitCode == 0 then nonEmpty(result.out.text().trim)
        else
          val vaultText = config.opVault.fold("")(vault => s" in vault $vault")
          Console.err.println(
            s"1Password field read failed for item '$item'$vaultText field '$field':",
          )
          Console.err.println(result.err.text())
          Console.err.println("Please run 'eval $(op signin)'")
          None
      }
      .headOption

  private def fieldNames(name: String): List[String] =
    val aliases =
      name match
        case "PGP_SECRET_KEY" =>
          List("PGP_SECRET", "PGP_SECRET_KEY", "PGP_PRIVATE_KEY", "PGP_KEY")
        case "PGP_SECRET_KEY_PASSWORD" =>
          List(
            "PGP_PASSPHRASE",
            "PGP_SECRET_KEY_PASSWORD",
            "PGP_SECRET_PASSWORD",
            "PGP_PASSWORD",
          )
        case "SONATYPE_USERNAME" =>
          List(
            "SONATYPE_USERNAME",
            "SONATYPE_USER",
            "CENTRAL_USERNAME",
            "CENTRAL_USER",
          )
        case "SONATYPE_PASSWORD" =>
          List(
            "SONATYPE_PASSWORD",
            "SONATYPE_TOKEN",
            "CENTRAL_PASSWORD",
            "CENTRAL_TOKEN",
          )
        case _ => Nil
    (aliases).distinct

  private def inferRepo(): Option[String] =
    val remote =
      os.proc("git", "remote", "get-url", "origin").call(check = false)
    if remote.exitCode != 0 then None
    else parseRemote(remote.out.text().trim)

  private def parseRemote(remote: String): Option[String] =
    val https = "^https://github\\.com/([^/]+/[^/.]+)(?:\\.git)?$".r
    val ssh = "^git@github\\.com:([^/]+/[^/.]+)(?:\\.git)?$".r
    remote match
      case https(repo) => Some(repo)
      case ssh(repo)   => Some(repo)
      case _           => None

  private def requireCommand(command: String): Unit =
    if os.proc("which", command).call(check = false).exitCode != 0 then
      Console.err.println(s"Missing required command: $command")
      sys.exit(1)

  private val ghNoPagerEnv: Map[String, String] =
    Map("GH_PAGER" -> "cat", "PAGER" -> "cat")

  private def env(name: String): Option[String] =
    sys.env.get(name).flatMap(nonEmpty)

  private def envList(name: String): List[String] =
    sys.env
      .get(name)
      .toList
      .flatMap(_.split(',').toList)
      .flatMap(nonEmpty)

  private def nonEmpty(value: String): Option[String] =
    Option(value.trim).filter(_.nonEmpty)

  private def missingValueMessage(name: String): String =
    s"Missing value for $name. Set $$$name, $$OP_${name}_REF, --op-item, --op-gpg-item/--op-sonatype-item, or put field '$name' in a fallback 1Password item such as 'SONATYPE_CREDS'. Use --verbose to show failed op reads."

  private def printUsage(): Unit =
    println(
      """Configure the GitHub repo for the Sonatype Central release workflow:
        |  - set the release secrets used by .github/workflows/publish.yml
        |  - enable vulnerability alerts / dependency graph
        |
        |Secret values are resolved, in order, from:
        |  1. a plain environment variable of the same name
        |  2. an explicit 1Password ref env var OP_<NAME>_REF
        |  3. a per-group 1Password item ref: --op-gpg-item (PGP_*) / --op-sonatype-item (SONATYPE_*)
        |  4. a single 1Password item ref via --op-item, deriving <ITEM_REF>/<NAME>
        |  5. an item title plus vault: --op-vault VAULT --op-item-name ITEM, deriving op://VAULT/ITEM/<NAME>
        |  6. default 1Password item-title fallback: SONATYPE_CREDS, GPG, scala-llm-template, scala-llm-template-release, sonatype, gpg,
        |     plus comma-separated OP_ITEM_NAMES or repeated --op-fallback-item ITEM
        |
        |Usage:
        |  scala-cli run scripts/setup-gh-repo.scala -- [--repo OWNER/REPO] [--op-item REF] [--op-gpg-item REF] [--op-sonatype-item REF]
        |  scala-cli run scripts/setup-gh-repo.scala -- --op-vault VAULT --op-item-name ITEM
        |
        |Examples:
        |  scala-cli run scripts/setup-gh-repo.scala -- --op-item op://Private/scala-llm-template-release
        |  scala-cli run scripts/setup-gh-repo.scala -- --op-vault Private --op-sonatype-item-name scala-llm-template-sonatype --op-gpg-item-name scala-llm-template-gpg
        |  scala-cli run scripts/setup-gh-repo.scala -- --op-vault Private --op-fallback-item scala-llm-template
        |""".stripMargin,
    )
