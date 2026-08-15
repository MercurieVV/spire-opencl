//> using scala 3.8.4
//> using dep org.scalameta::mdoc:2.9.0
//> using dep org.typelevel::laika-io:1.3.2

package docs

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.time.LocalDate
import java.util.regex.Matcher
import laika.ast.Path.Root
import laika.api.Transformer
import laika.format.HTML
import laika.format.Markdown
import laika.helium.Helium
import laika.helium.config.{HeliumIcon, IconLink, TextLink, ThemeNavigationSection}
import laika.io.syntax._

object DocsMain:

  private val SiteUrl = "https://mercurievv.github.io/spire-opencl/"
  private val RepoUrl = "https://github.com/MercurieVV/spire-opencl"

  private val Description =
    "Scala 3 GPU numeric computing with Typelevel Spire and OpenCL: write ordinary Spire typeclass code and compile it to OpenCL kernels."

  def main(args: Array[String]): Unit =
    val in = Paths.get(args.headOption.getOrElse("docs"))
    val out = Paths.get(args.lift(1).getOrElse("website/docs"))

    val theme = Helium.defaults.all
      .metadata(
        title       = Some("spire-opencl"),
        language    = Some("en"),
        authors     = Seq("Victor Mercurievv"),
        description = Some(Description),
      )
      .site
      .baseURL(SiteUrl)
      .site
      .topNavigationBar(
        navLinks = Seq(
          IconLink.external(RepoUrl, HeliumIcon.github),
        ),
      )
      .site
      .mainNavigation(
        prependLinks = Seq(
          ThemeNavigationSection("Documentation", TextLink.internal(Root / "README.md", "Start")),
        ),
      )
      .build

    Transformer
      .from(Markdown)
      .to(HTML)
      .using(Markdown.GitHubFlavor)
      .parallel[IO]
      .withTheme(theme)
      .build
      .use { transformer =>
        transformer
          .fromDirectory(in.toString)
          .toDirectory(out.toString)
          .transform
          .flatMap(_ =>
            IO.blocking {
              SiteDiscovery.write(out)
            },
          )
          .void
      }
      .unsafeRunSync()

  private object SiteDiscovery:

    private val Pages = Seq(
      "index.html"          -> "Start",
      "var-operations.html" -> "State and More Operations",
      "benchmarking.html"   -> "Benchmarking",
    )

    def write(out: java.nio.file.Path): Unit =
      writeRobots(out)
      writeSitemap(out)
      enrichHtml(out)

    private def writeRobots(out: java.nio.file.Path): Unit =
      val robots =
        s"""User-agent: *
           |Allow: /
           |
           |Sitemap: ${SiteUrl}sitemap.xml
           |""".stripMargin
      Files.writeString(out.resolve("robots.txt"), robots, StandardCharsets.UTF_8)

    private def writeSitemap(out: java.nio.file.Path): Unit =
      val today = LocalDate.now()
      val urls = Pages
        .filter((file, _) => Files.exists(out.resolve(file)))
        .map { case (file, _) =>
          val loc = if file == "index.html" then SiteUrl else s"$SiteUrl$file"
          s"""  <url>
             |    <loc>$loc</loc>
             |    <lastmod>$today</lastmod>
             |  </url>""".stripMargin
        }
        .mkString("\n")
      val sitemap =
        s"""<?xml version="1.0" encoding="UTF-8"?>
           |<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
           |$urls
           |</urlset>
           |""".stripMargin
      Files.writeString(out.resolve("sitemap.xml"), sitemap, StandardCharsets.UTF_8)

    private def enrichHtml(out: java.nio.file.Path): Unit =
      Pages.foreach { case (file, fallbackTitle) =>
        val path = out.resolve(file)
        if Files.exists(path) then
          val html = Files.readString(path, StandardCharsets.UTF_8)
          if !html.contains("""property="og:title"""") then
            val title = findBetween(html, "<title>", "</title>").getOrElse(fallbackTitle)
            val canonical = if file == "index.html" then SiteUrl else s"$SiteUrl$file"
            val canonicalTag =
              if html.contains("""rel="canonical"""") then ""
              else s"""  <link rel="canonical" href="$canonical"/>"""
            val tags =
              s"""
                 |$canonicalTag
                 |  <meta name="keywords" content="Scala, Scala 3, GPU computing, OpenCL, Typelevel, Spire, numeric programming, symbolic expressions, generated kernels"/>
                 |  <meta property="og:type" content="website"/>
                 |  <meta property="og:site_name" content="spire-opencl"/>
                 |  <meta property="og:title" content="${escapeAttr(title)}"/>
                 |  <meta property="og:description" content="${escapeAttr(Description)}"/>
                 |  <meta property="og:url" content="$canonical"/>
                 |  <meta name="twitter:card" content="summary"/>
                 |  <meta name="twitter:title" content="${escapeAttr(title)}"/>
                 |  <meta name="twitter:description" content="${escapeAttr(Description)}"/>
                 |""".stripMargin
            Files.writeString(
              path,
              html.replaceFirst("(?s)(\\s*<link rel=\"stylesheet\")", Matcher.quoteReplacement(tags) + "$1"),
              StandardCharsets.UTF_8,
            )
      }

    private def findBetween(input: String, start: String, end: String): Option[String] =
      val from = input.indexOf(start)
      if from < 0 then None
      else
        val valueStart = from + start.length
        val to = input.indexOf(end, valueStart)
        Option.when(to >= 0)(input.substring(valueStart, to))

    private def escapeAttr(value: String): String =
      value
        .replace("&", "&amp;")
        .replace("\"", "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
