package scdgs.top

import java.nio.file.Paths

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import scdgs.trace.{TraceFormat, TraceRecord}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Uses parsed fixture control records in a deterministic top-level bootstrap exercise. */
class FixtureReplaySpec extends AnyFunSuite with Matchers with ChiselSim {
  private val testParams = ScdGsParams(
    maxGaussians = 16,
    activeCapacity = 2,
    residentCapacity = 2,
    tileWidth = 2,
    renderQueueDepth = 4
  )

  private def word(value: String): BigInt = BigInt(value.stripPrefix("0x"), 16)
  private def first(records: Seq[TraceRecord], kind: String): TraceRecord =
    records.find(_.kind == kind).getOrElse(fail(s"fixture lacks $kind"))
  private def projection(records: Seq[TraceRecord], role: String): TraceRecord =
    records.find(record => record.kind == "PROJECTION" && record.fields("role") == role)
      .getOrElse(fail(s"fixture lacks $role PROJECTION"))

  test("public trace fixture drives a seeded state through canonical feedback") {
    val records = TraceFormat.read(Paths.get("..", "docs", "fixtures", "six_frame.scdgs-trace").toString)
    TraceFormat.validate(records, maxGaussians = 16)
    val update = first(records, "UPDATE").fields
    val priorSummary = projection(records, "prior").fields
    val currentView = projection(records, "current").fields
    val target = first(records, "TARGET").fields

    simulate(new ScdGsTop(testParams)) { dut =>
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.stateSeedIn.valid.poke(false.B)
      dut.io.candidateFillIn.valid.poke(false.B)
      dut.io.targetIn.valid.poke(false.B)
      dut.io.frameEnd.poke(false.B)
      dut.io.candidateReq.ready.poke(true.B)
      dut.io.metadataReadGid.poke(0.U)
      dut.clock.step()

      driveState(dut.io.stateSeedIn.bits.state, gid = update("gid").toInt)
      dut.io.stateSeedIn.bits.tier.poke(ScdGsConsts.TierActive.U)
      dut.io.stateSeedIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.stateSeedIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.stateSeedIn.valid.poke(false.B)

      driveUpdate(dut.io.canonicalIn.bits, gid = update("gid").toInt, frame = update("frame").toInt, strength = word(update("n")))
      driveProjection(
        dut.io.priorTrainingSummaryIn.bits,
        gid = priorSummary("gid").toInt,
        frame = priorSummary("frame").toInt,
        request = ScdGsConsts.RequestTraining,
        area = fpOne,
        tile = target("pixel").toInt,
        depth = fpTenth
      )
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 12)(dut.io.canonicalIn.ready.peek().litToBoolean && dut.io.priorTrainingSummaryIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)

      // The first FRU residency is candidate-tier even though StateSPM was
      // seeded locally. Complete the paper-required candidate refill before
      // pairing the current view at CIU.
      driveCandidateFill(dut.io.candidateFillIn.bits, gid = update("gid").toInt)
      dut.io.candidateFillIn.valid.poke(true.B)
      stepUntil(dut.clock, 200)(dut.io.candidateFillIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.candidateFillIn.valid.poke(false.B)

      driveProjection(
        dut.io.currentViewIn.bits,
        gid = currentView("gid").toInt,
        frame = currentView("frame").toInt,
        request = ScdGsConsts.RequestTraining,
        area = fpOne,
        tile = target("pixel").toInt,
        depth = fpTenth
      )
      dut.io.currentViewIn.valid.poke(true.B)
      stepUntil(dut.clock, 200)(dut.io.counters.ciuLight.peek().litValue == 1)
      dut.io.currentViewIn.valid.poke(false.B)

      // TUU must retain its one target before it accepts the trajectory. The
      // target therefore transfers before `frameEnd` lets TRU drain the
      // training record into the top-level trajectory bridge.
      val rgb = target("rgb").split(",").map(word)
      dut.io.targetIn.bits.pixel.poke(target("pixel").toInt.U)
      dut.io.targetIn.bits.frameId.poke(target("frame").toInt.U)
      rgb.zipWithIndex.foreach { case (value, index) => dut.io.targetIn.bits.rgb(index).poke(value.U) }
      dut.io.targetIn.valid.poke(true.B)
      dut.io.targetIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.targetIn.valid.poke(false.B)
      // The top-level handoff must not accept another target before the first
      // reaches TUU's sole target buffer.
      dut.io.targetIn.ready.expect(false.B)

      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      dut.io.targetIn.ready.expect(false.B)
      stepUntil(dut.clock, 280)(dut.io.counters.tuuUpdates.peek().litValue == 1)
      stepUntil(dut.clock, 200)(dut.io.counters.siuCommit.peek().litValue == 2)
      dut.io.counters.siuCommit.expect(2.U)
      dut.io.counters.tuuUpdates.expect(1.U)
    }
  }
}
