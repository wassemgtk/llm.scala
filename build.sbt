name := "llm-training"
version := "1.0"
scalaVersion := "2.13.8"

val scalaTestVersion = "3.2.12"
val catsVersion = "2.7.0"
val catsEffectVersion = "3.3.12"
val json4sVersion = "4.0.5"
val logbackVersion = "1.2.11"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % Test,
  "org.typelevel" %% "cats-core" % catsVersion,
  "org.typelevel" %% "cats-effect" % catsEffectVersion,
  "org.json4s" %% "json4s-jackson" % json4sVersion,
  "ch.qos.logback" % "logback-classic" % logbackVersion
)

scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-Ywarn-unused:imports",
  "-Ywarn-unused:locals",
  "-Ywarn-unused:privates",
  "-Ywarn-unused:params",
  "-Ywarn-unused:patvars",
  "-Ywarn-unused:implicits",
  "-Xlint:infer-any",
  "-Xlint:missing-interpolator",
  "-Xlint:doc-detached",
  "-Xlint:private-shadow",
  "-Xlint:type-parameter-shadow",
  "-Xlint:poly-implicit-overload",
  "-Xlint:option-implicit",
  "-Xlint:delayedinit-select",
  "-Xlint:package-object-classes",
  "-Xlint:stars-align",
  "-Xlint:constant",
  "-Xlint:nonlocal-return",
  "-Xlint:valpattern",
  "-Xlint:eta-zero",
  "-Xlint:eta-sam",
  "-Xlint:deprecation",
  "-Xlint:nullary-unit",
  "-Xlint:nullary-override",
  "-Xlint:unsound-match",
  "-Xlint:by-name-right-associative",
  "-Xfatal-warnings"
)

Test / testOptions += Tests.Argument(TestFrameworks.ScalaTest, "-oDF")

import sbtassembly.AssemblyPlugin.autoImport._
import sbtassembly.PathList
import sbtassembly.MergeStrategy

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}


Compile / run / mainClass := Some("llm.Main")