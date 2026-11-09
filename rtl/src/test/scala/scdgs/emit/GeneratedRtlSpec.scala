package scdgs.emit

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import circt.stage.ChiselStage
import scdgs.config.ScdGsParams
import scdgs.top.ScdGsPublicTop
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Emits SystemVerilog and verifies the public wrapper and five stages are present. */
class GeneratedRtlSpec extends AnyFunSuite with Matchers {
  private def deleteTree(path: Path): Unit = {
    val stream = Files.walk(path)
    try stream.sorted(java.util.Comparator.reverseOrder()).forEach(candidate => Files.deleteIfExists(candidate))
    finally stream.close()
  }

  test("SystemVerilog emission exposes the public top and all five mechanism stages") {
    val target = Files.createTempDirectory("scdgs-sv-")
    try {
      ChiselStage.emitSystemVerilogFile(
        new ScdGsPublicTop(
          ScdGsParams(
            maxGaussians = 16,
            activeCapacity = 2,
            residentCapacity = 2,
            tileWidth = 2,
            renderQueueDepth = 4
          )
        ),
        args = Array("--target-dir", target.toString),
        firtoolOpts = Array(
          "--split-verilog",
          "--lowering-options=disallowPackedArrays,disallowLocalVariables",
          "--disable-all-randomization",
          "--strip-debug-info"
        )
      )
      val stream = Files.walk(target)
      val contents = try {
        import scala.jdk.CollectionConverters._
        stream.iterator().asScala
          .filter(path => path.getFileName.toString.endsWith(".sv"))
          .map(path => Files.readString(path, StandardCharsets.UTF_8))
          .mkString("\n")
      } finally stream.close()
      contents should include("module ScdGsPublicTop")
      Seq("SplatImpactUnit", "FusedResidencyUnit", "ConditionedIssueUnit", "TieredRenderUnit", "TrainingUpdateUnit").foreach { name =>
        contents should include(s"module $name")
      }
      contents should include("canonicalIn")
      contents should include("candidateReq")
    } finally {
      deleteTree(target)
    }
  }
}
