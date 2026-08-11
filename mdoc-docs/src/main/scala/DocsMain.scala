//> using scala 3.3.4
//> using dep org.scalameta::mdoc:2.9.0

package docs

import java.nio.file.Paths

object DocsMain:

  def main(args: Array[String]): Unit =
    val settings = mdoc
      .MainSettings()
      .withIn(Paths.get("docs"))
      .withOut(Paths.get("website", "docs"))
      .withClasspath(System.getProperty("java.class.path"))
      .withArgs(args.toList)
    val exitCode = mdoc.Main.process(settings)
    if exitCode != 0 then sys.exit(exitCode)
