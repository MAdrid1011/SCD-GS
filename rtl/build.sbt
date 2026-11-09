ThisBuild / scalaVersion := "2.13.18"
ThisBuild / organization := "org.scdgs"
ThisBuild / version := "0.1.0"

val chiselVersion = sys.props.getOrElse("chiselVersion", "7.11.0")

lazy val root = (project in file("."))
  .settings(
    name := "scdgs-rtl",
    Compile / unmanagedSourceDirectories += baseDirectory.value / "vendor/berkeley-hardfloat/hardfloat/src/main/scala",
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "org.scalatest" %% "scalatest" % "3.2.17" % Test
    ),
    Compile / mainClass := Some("scdgs.emit.EmitScdGsTop"),
    Test / parallelExecution := false,
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-language:reflectiveCalls"
    )
  )
