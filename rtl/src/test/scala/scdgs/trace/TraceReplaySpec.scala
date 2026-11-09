package scdgs.trace

import java.nio.file.Paths
import scala.io.Source

import scdgs.config.ScdGsParams
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests the full public trace grammar and its six-frame fixture. */
class TraceReplaySpec extends AnyFunSuite with Matchers {
  private def fixture: Seq[TraceRecord] =
    TraceFormat.read(Paths.get("..", "docs", "fixtures", "six_frame.scdgs-trace").toString)

  private def width(record: TraceRecord, field: String): Int =
    record.fields(field).split(",", -1).length

  private def first(records: Seq[TraceRecord], kind: String): TraceRecord =
    records.find(_.kind == kind).getOrElse(fail(s"fixture lacks $kind"))

  private def fixtureLine(kind: String): String = {
    val path = Paths.get("..", "docs", "fixtures", "six_frame.scdgs-trace").toString
    val source = Source.fromFile(path)
    try source.getLines().find(_.startsWith(s"$kind|"))
      .getOrElse(fail(s"fixture lacks $kind line"))
    finally source.close()
  }

  test("public six-frame trace carries complete state, projection, and feedback bundles") {
    val records = fixture
    TraceFormat.validate(records, maxGaussians = 16)

    records.count(_.kind == "FRAME") shouldBe 12
    records.count(_.kind == "STATE") shouldBe 1
    records.count(_.kind == "UPDATE") shouldBe 5
    records.count(record => record.kind == "EVENT" && record.fields("source") == "commit") shouldBe 5
    records.exists(record => record.kind == "EVENT" && record.fields("source") == "confirmation") shouldBe false
    records.exists(record => record.kind == "EVENT" && record.fields("source") == "render-read") shouldBe true
    records.exists(record => record.kind == "REQUEST" && record.fields("kind") == "training") shouldBe true
    records.exists(_.kind == "TRAJECTORY") shouldBe true
    records.exists(_.kind == "TARGET") shouldBe true

    val state = first(records, "STATE")
    width(state, "mean") shouldBe 3
    width(state, "covariance") shouldBe 6
    width(state, "sh") shouldBe 48

    val projection = first(records, "PROJECTION")
    width(projection, "r_cam") shouldBe 9
    width(projection, "j_pi") shouldBe 6
    width(projection, "j_cov") shouldBe 6
    width(projection, "view_direction") shouldBe 3
    width(projection, "sh_basis") shouldBe 16
    width(projection, "center") shouldBe 2
    width(projection, "conic") shouldBe 3
    width(projection, "view_color") shouldBe 3

    first(records, "UPDATE").fields("dsh").split(",", -1).length shouldBe 48
    width(first(records, "TRAJECTORY"), "color") shouldBe 3
  }

  test("trace parser rejects an unsupported record kind") {
    an[IllegalArgumentException] should be thrownBy TraceFormat.parseLine("RESULT|value=1", 1)
  }

  test("trace parser rejects missing camera data, duplicate fields, and unknown fields") {
    val missingCamera = fixtureLine("PROJECTION").replaceFirst("\\|r_cam=[^|]+", "")
    an[IllegalArgumentException] should be thrownBy TraceFormat.parseLine(missingCamera, 1)

    an[IllegalArgumentException] should be thrownBy TraceFormat.parseLine("FRAME|id=0|edge=begin|id=0", 1)
    an[IllegalArgumentException] should be thrownBy TraceFormat.parseLine("FRAME|id=0|edge=begin|extra=yes", 1)
  }

  test("trace validator rejects malformed binary32 fields, a truncated SH vector, and a mismatched request") {
    val malformedBinary32 = fixture.map { record =>
      if (record.kind == "UPDATE" && record.fields("frame") == "1") {
        record.copy(fields = record.fields.updated("n", "not-a-binary32-word"))
      } else record
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(malformedBinary32, maxGaussians = 16)

    val shortSh = fixture.map { record =>
      if (record.kind == "UPDATE" && record.fields("frame") == "1") {
        record.copy(fields = record.fields.updated("dsh", List.fill(47)("0x0").mkString(",")))
      } else record
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(shortSh, maxGaussians = 16)

    val mismatchedRequest = fixture.map { record =>
      if (record.kind == "REQUEST") record.copy(fields = record.fields.updated("kind", "user"))
      else record
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(mismatchedRequest, maxGaussians = 16)

    val zeroRadius = fixture.map { record =>
      if (record.kind == "PROJECTION") record.copy(fields = record.fields.updated("radius", "0x00000000"))
      else record
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(zeroRadius, maxGaussians = 16)
  }

  test("trace validator rejects an invalid frame sequence") {
    val invalidSequence = fixture.map { record =>
      if (record.kind == "FRAME" && record.fields("id") == "1" && record.fields("edge") == "begin") {
        record.copy(fields = record.fields.updated("id", "0"))
      } else record
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(invalidSequence, maxGaussians = 16)
  }

  test("trace validator bounds frame identifiers and preserves terminal feedback causality") {
    val wideBoundary = fixture.map { record =>
      if (record.kind == "FRAME" && record.fields("id") == "0" && record.fields("edge") == "begin") {
        record.copy(fields = record.fields.updated("id", "65536"))
      } else record
    }
    val boundaryError = the[IllegalArgumentException] thrownBy TraceFormat.validate(wideBoundary, maxGaussians = 16)
    boundaryError.getMessage should include("exceeds the 16-bit public frame range")

    val wideTarget = fixture.map { record =>
      if (record.kind == "TARGET") record.copy(fields = record.fields.updated("frame", "65536")) else record
    }
    val targetError = the[IllegalArgumentException] thrownBy TraceFormat.validate(wideTarget, maxGaussians = 16)
    targetError.getMessage should include("exceeds the 16-bit public frame range")

    val terminalTrainingFrame = fixture.filter { record =>
      if (record.kind == "FRAME") record.fields("id") == "0" || record.fields("id") == "1"
      else record.fields("frame") == "0" || record.fields("frame") == "1"
    }.map { record =>
      if (record.kind == "FRAME" && record.fields("id") == "1") {
        record.copy(fields = record.fields.updated("id", "65535"))
      } else if (record.fields.get("frame").contains("1")) {
        record.copy(fields = record.fields.updated("frame", "65535"))
      } else record
    }
    val terminalError = the[IllegalArgumentException] thrownBy TraceFormat.validate(terminalTrainingFrame, maxGaussians = 16)
    terminalError.getMessage should include("TRAJECTORY must use a frame below 65535")
  }

  test("trace validator bounds tile and TRU blend-order fields to the public interface") {
    val wideTile = fixture.map { record =>
      if (record.kind == "PROJECTION") record.copy(fields = record.fields.updated("tile", "4096")) else record
    }
    val tileError = the[IllegalArgumentException] thrownBy TraceFormat.validate(wideTile, maxGaussians = 16)
    tileError.getMessage should include("exceeds the 12-bit public tile range")

    val wideBlend = fixture.map { record =>
      if (record.kind == "TRAJECTORY") record.copy(fields = record.fields.updated("blend", "64")) else record
    }
    val blendError = the[IllegalArgumentException] thrownBy TraceFormat.validate(wideBlend, maxGaussians = 16)
    blendError.getMessage should include("exceeds the public TRU blend-order range [0, 63]")
  }

  test("trace validator rejects a non-finite projection depth") {
    Seq("0x7fc00000", "0x7f800000").foreach { depth =>
      val nonFiniteDepth = fixture.map { record =>
        if (record.kind == "PROJECTION") record.copy(fields = record.fields.updated("depth", depth)) else record
      }
      val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(nonFiniteDepth, maxGaussians = 16)
      error.getMessage should include("depth must be a finite binary32 value")
    }
  }

  test("trace validator rejects a repeated source-frame update for one Gaussian") {
    val repeatedUpdate = fixture.flatMap { record =>
      if (record.kind == "UPDATE" && record.fields("frame") == "1") Seq(record, record) else Seq(record)
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(repeatedUpdate, maxGaussians = 16)
  }

  test("trace validator rejects a commit without its matching canonical update") {
    val noMatchingUpdate = fixture.filterNot(record => record.kind == "UPDATE" && record.fields("frame") == "1")
    val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(noMatchingUpdate, maxGaussians = 16)
    error.getMessage should include("commit for gid 7, frame 1 has no preceding UPDATE")
  }

  test("trace validator requires one SIU propagation and preserves mandatory commits") {
    val missingPropagation = fixture.filterNot { record =>
      record.kind == "EVENT" && record.fields("source") == "commit" && record.fields("frame") == "2"
    }
    val missingError = the[IllegalArgumentException] thrownBy TraceFormat.validate(missingPropagation, maxGaussians = 16)
    missingError.getMessage should include("trace omits SIU propagation EVENT for gid 7, frame 2")

    val warmupConfirmation = fixture.map { record =>
      if (record.kind == "EVENT" && record.fields("source") == "commit" && record.fields("frame") == "2") {
        record.copy(fields = record.fields.updated("source", "confirmation"))
      } else record
    }
    val warmupError = the[IllegalArgumentException] thrownBy TraceFormat.validate(warmupConfirmation, maxGaussians = 16)
    warmupError.getMessage should include("confirmation for gid 7, frame 2 violates SIU's required commit")

    val structuralConfirmation = fixture.map { record =>
      if (record.kind == "EVENT" && record.fields("source") == "commit" && record.fields("frame") == "5") {
        record.copy(fields = record.fields.updated("source", "confirmation"))
      } else record
    }
    val structuralError = the[IllegalArgumentException] thrownBy TraceFormat.validate(structuralConfirmation, maxGaussians = 16)
    structuralError.getMessage should include("confirmation for gid 7, frame 5 violates SIU's required commit")

    val allowedConfirmation = fixture.map { record =>
      if (record.kind == "UPDATE" && record.fields("frame") == "5") {
        record.copy(fields = record.fields.updated("struct", "none"))
      } else if (record.kind == "EVENT" && record.fields("source") == "commit" && record.fields("frame") == "5") {
        record.copy(fields = record.fields.updated("source", "confirmation"))
      } else record
    }
    noException should be thrownBy TraceFormat.validate(allowedConfirmation, maxGaussians = 16)
  }

  test("trace validator prevents CIU views from bypassing a commit") {
    val noCommit = fixture.filterNot(record => record.kind == "EVENT" && record.fields("source") == "commit")
    val currentViewError = the[IllegalArgumentException] thrownBy TraceFormat.validate(noCommit, maxGaussians = 16)
    currentViewError.getMessage should include("current PROJECTION for gid 7, frame 1 has no preceding commit")
  }

  test("trace validator rejects duplicate stage-handoff records for one source frame") {
    val duplicateCases = Seq(
      ((record: TraceRecord) => record.kind == "STATE", "duplicate STATE"),
      ((record: TraceRecord) => record.kind == "PROJECTION" && record.fields("role") == "current", "duplicate current PROJECTION"),
      ((record: TraceRecord) => record.kind == "EVENT" && record.fields("source") == "commit", "duplicate commit"),
      ((record: TraceRecord) => record.kind == "REQUEST", "duplicate REQUEST"),
      ((record: TraceRecord) => record.kind == "TRAJECTORY", "duplicate TRAJECTORY"),
      ((record: TraceRecord) => record.kind == "EVENT" && record.fields("source") == "render-read", "duplicate render-read")
    )
    duplicateCases.foreach { case (matches, expected) =>
      val records = fixture.flatMap(record => if (matches(record)) Seq(record, record) else Seq(record))
      val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(records, maxGaussians = 16)
      error.getMessage should include(expected)
    }
  }

  test("trace validator requires updates to use a strictly prior training summary") {
    val userProjection = fixture.map { record =>
      if (record.kind == "PROJECTION" || record.kind == "REQUEST") {
        record.copy(fields = record.fields.updated(if (record.kind == "PROJECTION") "request" else "kind", "user"))
      } else record
    }
    val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(userProjection, maxGaussians = 16)
    error.getMessage should include("prior PROJECTION must request training")
  }

  test("trace validator requires one consumed bootstrap prior and one request per current view") {
    val bootstrap = fixture.find(record => record.kind == "PROJECTION" && record.fields("role") == "prior")
      .getOrElse(fail("fixture lacks prior PROJECTION"))
    val duplicateBootstrap = fixture.flatMap { record =>
      if (record == bootstrap) Seq(record, record) else Seq(record)
    }
    val duplicateError = the[IllegalArgumentException] thrownBy TraceFormat.validate(duplicateBootstrap, maxGaussians = 16)
    duplicateError.getMessage should include("duplicate bootstrap prior PROJECTION")

    val orphanBootstrap = fixture.flatMap { record =>
      if (record == bootstrap) Seq(record, record.copy(fields = record.fields.updated("gid", "8"))) else Seq(record)
    }
    val orphanError = the[IllegalArgumentException] thrownBy TraceFormat.validate(orphanBootstrap, maxGaussians = 16)
    orphanError.getMessage should include("bootstrap prior PROJECTION for gid 8 has no matching first UPDATE")

    val missingRequest = fixture.filterNot(_.kind == "REQUEST")
    val requestError = the[IllegalArgumentException] thrownBy TraceFormat.validate(missingRequest, maxGaussians = 16)
    requestError.getMessage should include("current PROJECTION for gid 7, frame 1 has no matching REQUEST")
  }

  test("trace validator requires trajectories to use a same-frame training current projection") {
    val templateProjection = fixture.find(record => record.kind == "PROJECTION" && record.fields("role") == "current")
      .getOrElse(fail("fixture lacks current PROJECTION"))
    val templateTrajectory = first(fixture, "TRAJECTORY")
    val userProjection = templateProjection.copy(fields = templateProjection.fields.updated("frame", "1").updated("request", "user"))
    val userRequest = TraceRecord("REQUEST", Map("gid" -> "7", "frame" -> "1", "kind" -> "user"))
    val userTrajectory = templateTrajectory.copy(fields = templateTrajectory.fields.updated("frame", "1"))
    val records = fixture.flatMap { record =>
      if (record.kind == "EVENT" && record.fields("source") == "commit" && record.fields("frame") == "1") {
        Seq(record, userProjection, userRequest, userTrajectory)
      } else Seq(record)
    }
    val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(records, maxGaussians = 16)
    error.getMessage should include("TRAJECTORY for gid 7 requires a same-frame training current PROJECTION")
  }

  test("trace validator rejects a filter trajectory") {
    val filterTrajectory = fixture.map { record =>
      if (record.kind == "TRAJECTORY") record.copy(fields = record.fields.updated("path", "filter")) else record
    }
    val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(filterTrajectory, maxGaussians = 16)
    error.getMessage should include("TRAJECTORY path must be full or light")
  }

  test("trace validator requires a render-read to match its training trajectory source frame") {
    val noTrainingTrajectory = fixture.filterNot(_.kind == "TRAJECTORY")
    val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(noTrainingTrajectory, maxGaussians = 16)
    error.getMessage should include("render-read for gid 7, frame 1 has no matching training TRAJECTORY")

    val missingRead = fixture.filterNot(record => record.kind == "EVENT" && record.fields("source") == "render-read")
    val missingError = the[IllegalArgumentException] thrownBy TraceFormat.validate(missingRead, maxGaussians = 16)
    missingError.getMessage should include("trace omits render-read for training TRAJECTORY gid 7, frame 1")

    val delayedRead = fixture.flatMap { record =>
      if (record.kind == "EVENT" && record.fields("source") == "render-read") Seq.empty
      else if (record.kind == "FRAME" && record.fields("id") == "2" && record.fields("edge") == "begin") {
        Seq(record, TraceRecord("EVENT", Map("gid" -> "7", "frame" -> "2", "source" -> "render-read")))
      } else Seq(record)
    }
    val delayedError = the[IllegalArgumentException] thrownBy TraceFormat.validate(delayedRead, maxGaussians = 16)
    delayedError.getMessage should include("render-read for gid 7, frame 2 has no matching training TRAJECTORY")
  }

  test("trace validator requires one target for each training trajectory") {
    val missingTarget = fixture.filterNot(_.kind == "TARGET")
    val missingError = the[IllegalArgumentException] thrownBy TraceFormat.validate(missingTarget, maxGaussians = 16)
    missingError.getMessage should include("trace omits TARGET for training TRAJECTORY pixel 3, frame 1")

    val orphanTarget = fixture.flatMap { record =>
      if (record.kind == "TARGET") Seq(record, record) else Seq(record)
    }
    val orphanError = the[IllegalArgumentException] thrownBy TraceFormat.validate(orphanTarget, maxGaussians = 16)
    orphanError.getMessage should include("trace has TARGET for pixel 3, frame 1 without a matching training TRAJECTORY")
  }

  test("trace validator uses the current tile as the public representative-pixel key") {
    val mismatchedPixel = fixture.map { record =>
      if (record.kind == "TRAJECTORY") record.copy(fields = record.fields.updated("pixel", "4")) else record
    }
    val error = the[IllegalArgumentException] thrownBy TraceFormat.validate(mismatchedPixel, maxGaussians = 16)
    error.getMessage should include("TRAJECTORY pixel must equal its current PROJECTION tile")
  }

  test("trace validator rejects a gid outside the supplied public configuration") {
    val invalid = fixture.map { record =>
      if (record.kind == "STATE") record.copy(fields = record.fields.updated("gid", "16"))
      else record
    }
    an[IllegalArgumentException] should be thrownBy TraceFormat.validate(invalid, maxGaussians = 16)
  }

  test("replay command uses the default public Gaussian-ID domain") {
    ReplayTrace.defaultMaxGaussians shouldBe ScdGsParams().maxGaussians
    val defaultDomainGid = 128
    val records = fixture.map { record =>
      if (record.fields.contains("gid")) record.copy(fields = record.fields.updated("gid", defaultDomainGid.toString))
      else record
    }
    noException should be thrownBy TraceFormat.validate(records, maxGaussians = ReplayTrace.defaultMaxGaussians)
  }
}
