package scdgs.top

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Integration test for five-stage wiring, candidate refill, frame edge, and queue backpressure. */
class ScdGsLoopSpec extends AnyFunSuite with Matchers with ChiselSim {
  private val testParams = ScdGsParams(
    maxGaussians = 16,
    activeCapacity = 2,
    residentCapacity = 2,
    tileWidth = 2,
    renderQueueDepth = 4
  )

  test("ScdGsTop rejects a user projection as an SIU bootstrap summary") {
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

      driveState(dut.io.stateSeedIn.bits.state, gid = 7)
      dut.io.stateSeedIn.bits.tier.poke(ScdGsConsts.TierActive.U)
      dut.io.stateSeedIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.stateSeedIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.stateSeedIn.valid.poke(false.B)

      driveUpdate(dut.io.canonicalIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorTrainingSummaryIn.bits, gid = 7, frame = 0, request = ScdGsConsts.RequestUser)
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.canonicalIn.ready.peek().litToBoolean && dut.io.priorTrainingSummaryIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.stateSeedIn.valid.poke(false.B)
      dut.io.candidateFillIn.valid.poke(false.B)
      stepUntil(dut.clock, 12)(dut.io.counters.invalidInput.peek().litValue == 1)
      dut.io.counters.siuCommit.expect(0.U)
      dut.io.counters.ciuFull.expect(0.U)
      dut.io.counters.ciuLight.expect(0.U)
      dut.io.counters.ciuFilter.expect(0.U)
    }
  }

  test("ScdGsTop counts an invalid canonical identifier without indexing state") {
    simulate(new ScdGsTop(testParams)) { dut =>
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.targetIn.valid.poke(false.B)
      dut.io.frameEnd.poke(false.B)
      dut.io.candidateReq.ready.poke(true.B)
      dut.io.metadataReadGid.poke(0.U)
      dut.clock.step()

      driveUpdate(dut.io.canonicalIn.bits, gid = 31, frame = 1)
      driveProjection(dut.io.priorTrainingSummaryIn.bits, gid = 31, frame = 0)
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 8)(dut.io.canonicalIn.ready.peek().litToBoolean && dut.io.priorTrainingSummaryIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)
      stepUntil(dut.clock, 120)(dut.io.counters.invalidInput.peek().litValue > 0)
      dut.io.counters.invalidInput.peek().litValue should be > BigInt(0)
    }
  }

  test("ScdGsTop accepts an external prior summary only for bootstrap") {
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

      driveState(dut.io.stateSeedIn.bits.state, gid = 7)
      dut.io.stateSeedIn.bits.tier.poke(ScdGsConsts.TierActive.U)
      dut.io.stateSeedIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.stateSeedIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.stateSeedIn.valid.poke(false.B)

      // The first external pair seeds the only accepted prior summary.
      driveUpdate(dut.io.canonicalIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorTrainingSummaryIn.bits, gid = 7, frame = 0, request = ScdGsConsts.RequestTraining)
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 12)(dut.io.canonicalIn.ready.peek().litToBoolean && dut.io.priorTrainingSummaryIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)

      // The user-tagged record would make SIU reject the second update if it
      // were consumed. It must remain unaccepted while SIU uses SplatSPM's
      // retained training summary from the bootstrap transaction.
      driveUpdate(dut.io.canonicalIn.bits, gid = 7, frame = 2)
      driveProjection(dut.io.priorTrainingSummaryIn.bits, gid = 7, frame = 1, request = ScdGsConsts.RequestUser)
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 200)(dut.io.canonicalIn.ready.peek().litToBoolean)
      dut.io.priorTrainingSummaryIn.ready.expect(false.B)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)
      dut.clock.step()
      dut.io.counters.invalidInput.expect(0.U)
    }
  }

  test("ScdGsTop holds a cold candidate until its matching StateSPM fill arrives") {
    simulate(new ScdGsTop(testParams)) { dut =>
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.stateSeedIn.valid.poke(false.B)
      dut.io.candidateFillIn.valid.poke(false.B)
      dut.io.targetIn.valid.poke(false.B)
      dut.io.frameEnd.poke(false.B)
      dut.io.candidateReq.ready.poke(false.B)
      dut.io.metadataReadGid.poke(0.U)
      dut.clock.step()

      driveUpdate(dut.io.canonicalIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorTrainingSummaryIn.bits, gid = 7, frame = 0, request = ScdGsConsts.RequestTraining)
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 12)(dut.io.canonicalIn.ready.peek().litToBoolean && dut.io.priorTrainingSummaryIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)

      stepUntil(dut.clock, 200)(dut.io.candidateReq.valid.peek().litToBoolean)
      dut.io.candidateReq.bits.gid.expect(7.U)
      dut.io.candidateReq.bits.bytes.expect(64.U)
      dut.io.counters.ciuFull.expect(0.U)
      dut.io.counters.ciuLight.expect(0.U)
      dut.io.counters.ciuFilter.expect(0.U)
      dut.clock.step(4)
      dut.io.counters.ciuFull.expect(0.U)
      dut.io.counters.ciuLight.expect(0.U)
      dut.io.counters.ciuFilter.expect(0.U)

      // The cold transaction must first hand off its aligned request. A fill
      // is not accepted until that request has actually transferred.
      dut.io.candidateReq.ready.poke(true.B)
      dut.clock.step()
      dut.io.counters.fruCandidateRefill.expect(1.U)

      driveCandidateFill(dut.io.candidateFillIn.bits, gid = 7)
      dut.io.candidateFillIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.candidateFillIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.candidateFillIn.valid.poke(false.B)

      driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = ScdGsConsts.RequestTraining)
      dut.io.currentViewIn.valid.poke(true.B)

      stepUntil(dut.clock, 200)(dut.io.counters.ciuLight.peek().litValue == 1)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.counters.ciuFull.expect(0.U)
      dut.io.counters.ciuFilter.expect(0.U)
    }
  }

  test("ScdGsTop carries a current user view to CIU filter work") {
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

      driveState(dut.io.stateSeedIn.bits.state, gid = 7)
      dut.io.stateSeedIn.bits.tier.poke(ScdGsConsts.TierActive.U)
      dut.io.stateSeedIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.stateSeedIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.stateSeedIn.valid.poke(false.B)

      driveUpdate(dut.io.canonicalIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorTrainingSummaryIn.bits, gid = 7, frame = 0, request = ScdGsConsts.RequestTraining)
      dut.io.canonicalIn.valid.poke(true.B)
      dut.io.priorTrainingSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 12)(dut.io.canonicalIn.ready.peek().litToBoolean && dut.io.priorTrainingSummaryIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.canonicalIn.valid.poke(false.B)
      dut.io.priorTrainingSummaryIn.valid.poke(false.B)

      // StateSeed does not allocate an FRU tier-directory entry. The first
      // commit is therefore candidate-resident and must complete its explicit
      // refill before CIU can consume the current user view.
      driveCandidateFill(dut.io.candidateFillIn.bits, gid = 7)
      dut.io.candidateFillIn.valid.poke(true.B)
      stepUntil(dut.clock, 200)(dut.io.candidateFillIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.candidateFillIn.valid.poke(false.B)

      driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = ScdGsConsts.RequestUser)
      dut.io.currentViewIn.valid.poke(true.B)
      stepUntil(dut.clock, 200)(dut.io.counters.ciuFilter.peek().litValue == 1)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.counters.ciuFull.expect(0.U)
      dut.io.counters.ciuLight.expect(0.U)
    }
  }
}
