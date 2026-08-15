//> using scala 3.8.4
//> using dep org.scalameta::mdoc:2.9.0
//> using dep org.typelevel::laika-io:1.3.2

package docs

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import java.nio.file.Paths
import laika.ast.Path.Root
import laika.api.Transformer
import laika.format.HTML
import laika.format.Markdown
import laika.helium.Helium
import laika.helium.config.{HeliumIcon, IconLink, TextLink, ThemeNavigationSection}
import laika.io.syntax._

object DocsMain:

  def main(args: Array[String]): Unit =
    val in = Paths.get(args.headOption.getOrElse("docs"))
    val out = Paths.get(args.lift(1).getOrElse("website/docs"))

    val theme = Helium.defaults.all
      .metadata(
        title    = Some("spire-opencl"),
        language = Some("en"),
      )
      .site
      .topNavigationBar(
        navLinks = Seq(
          IconLink.external("https://github.com/MercurieVV/spire-opencl", HeliumIcon.github),
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
          .void
      }
      .unsafeRunSync()
