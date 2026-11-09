package scdgs.tuu

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests target buffering, scalar-surrogate state write, and feedback. */
class TuuSpec extends AnyFunSuite with Matchers with ChiselSim {
  test("TUU pairs a buffered target with a trace into a full Adam-backed update") {
    simulate(new TrainingUpdateUnit(ScdGsParams(maxGaussians = 16, trainingLanes = 16))) { dut =>
      dut.io.traceIn.valid.poke(false.B)
      dut.io.targetIn.valid.poke(false.B)
      dut.io.updateOut.ready.poke(false.B)
      dut.io.stateWrite.ready.poke(false.B)
      dut.io.optimizerValid.poke(true.B)
      driveOptimizerState(dut.io.optimizerState, gid = 7)
      dut.clock.step()

      dut.io.targetIn.bits.pixel.poke(3.U)
      dut.io.targetIn.bits.frameId.poke(4.U)
      dut.io.targetIn.bits.rgb(0).poke(fpTenth.U)
      dut.io.targetIn.bits.rgb(1).poke(fpTenth.U)
      dut.io.targetIn.bits.rgb(2).poke(fpTenth.U)
      dut.io.targetIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.targetIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.targetIn.valid.poke(false.B)

      dut.io.traceIn.bits.gid.poke(7.U)
      dut.io.traceIn.bits.frameId.poke(4.U)
      dut.io.traceIn.bits.pixel.poke(3.U)
      dut.io.traceIn.bits.blendOrder.poke(0.U)
      dut.io.traceIn.bits.coverage.poke(fpHalf.U)
      dut.io.traceIn.bits.transmittance.poke(fpHalf.U)
      (0 until 3).foreach { channel => dut.io.traceIn.bits.color(channel).poke(fpHalf.U) }
      driveProjection(dut.io.traceIn.bits.projection, gid = 7, frame = 4)
      dut.io.traceIn.bits.path.poke(ScdGsConsts.PathLight.U)
      dut.io.traceIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.traceIn.valid.poke(false.B)

      stepUntil(dut.clock, 180)(dut.io.stateWrite.valid.peek().litToBoolean)
      // The feedback record is intentionally withheld until its StateSPM
      // update transfers. This prevents a feedback event from observing the
      // pre-update Gaussian under state-write backpressure.
      dut.io.updateOut.valid.expect(false.B)
      dut.io.stateWrite.bits.state.gid.expect(7.U)
      dut.io.stateWrite.bits.adamMoment(0).peek().litValue should be > BigInt(0)
      dut.io.stateWrite.bits.adamVariance(0).peek().litValue should be > BigInt(0)
      dut.io.stateWrite.ready.poke(true.B)
      dut.clock.step()
      dut.io.stateWrite.valid.expect(false.B)
      dut.io.updateOut.valid.expect(true.B)
      dut.io.updateOut.bits.gid.expect(7.U)
      dut.io.updateOut.bits.frameId.expect(5.U)
      dut.io.updateOut.bits.structural.expect(ScdGsConsts.StructuralNone.U)
      dut.io.updateOut.ready.poke(true.B)
      dut.clock.step()
      dut.io.updateCount.expect(1.U)
    }
  }

  test("TUU retains its target after rejecting non-training and mismatched traces") {
    simulate(new TrainingUpdateUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.traceIn.valid.poke(false.B)
      dut.io.targetIn.valid.poke(false.B)
      dut.io.updateOut.ready.poke(true.B)
      dut.io.stateWrite.ready.poke(true.B)
      dut.io.optimizerValid.poke(true.B)
      driveOptimizerState(dut.io.optimizerState, gid = 7)
      dut.clock.step()
      dut.io.targetIn.bits.pixel.poke(1.U)
      dut.io.targetIn.bits.frameId.poke(0.U)
      (0 until 3).foreach { channel => dut.io.targetIn.bits.rgb(channel).poke(fpZero.U) }
      dut.io.targetIn.valid.poke(true.B)
      dut.clock.step()
      dut.io.targetIn.valid.poke(false.B)
      dut.io.traceIn.bits.gid.poke(7.U)
      dut.io.traceIn.bits.frameId.poke(0.U)
      // A filter tag is not a training trajectory. TUU must reject it and
      // preserve the one buffered target for a later valid trajectory.
      dut.io.traceIn.bits.pixel.poke(1.U)
      dut.io.traceIn.bits.blendOrder.poke(0.U)
      dut.io.traceIn.bits.coverage.poke(fpHalf.U)
      dut.io.traceIn.bits.transmittance.poke(fpHalf.U)
      (0 until 3).foreach { channel => dut.io.traceIn.bits.color(channel).poke(fpHalf.U) }
      driveProjection(dut.io.traceIn.bits.projection, gid = 7, frame = 0)
      dut.io.traceIn.bits.path.poke(ScdGsConsts.PathFilter.U)
      dut.io.traceIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.traceIn.valid.poke(false.B)
      dut.io.invalidPairCount.expect(1.U)
      dut.io.updateOut.valid.expect(false.B)

      // A user-view trace is not a training trajectory even on a full/light
      // path, so it must not consume the target or create feedback.
      dut.io.traceIn.bits.path.poke(ScdGsConsts.PathLight.U)
      dut.io.traceIn.bits.projection.requestKind.poke(ScdGsConsts.RequestUser.U)
      dut.io.traceIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.traceIn.valid.poke(false.B)
      dut.io.invalidPairCount.expect(2.U)

      // A key mismatch is independently rejected and leaves the same target.
      dut.io.traceIn.bits.pixel.poke(2.U)
      dut.io.traceIn.bits.path.poke(ScdGsConsts.PathLight.U)
      dut.io.traceIn.bits.projection.requestKind.poke(ScdGsConsts.RequestTraining.U)
      dut.io.traceIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.traceIn.valid.poke(false.B)
      dut.io.invalidPairCount.expect(3.U)

      dut.io.traceIn.bits.pixel.poke(1.U)
      dut.io.traceIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.traceIn.valid.poke(false.B)
      stepUntil(dut.clock, 180)(dut.io.updateOut.valid.peek().litToBoolean)
      dut.io.updateOut.bits.gid.expect(7.U)
    }
  }

  test("TUU rejects a terminal-frame trajectory without wrapping feedback") {
    simulate(new TrainingUpdateUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      val terminalFrame = (BigInt(1) << 16) - 1
      dut.io.traceIn.valid.poke(false.B)
      dut.io.targetIn.valid.poke(false.B)
      dut.io.updateOut.ready.poke(true.B)
      dut.io.stateWrite.ready.poke(true.B)
      dut.io.optimizerValid.poke(true.B)
      driveOptimizerState(dut.io.optimizerState, gid = 7)
      dut.clock.step()

      dut.io.targetIn.bits.pixel.poke(3.U)
      dut.io.targetIn.bits.frameId.poke(terminalFrame.U)
      (0 until 3).foreach { channel => dut.io.targetIn.bits.rgb(channel).poke(fpTenth.U) }
      dut.io.targetIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.targetIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.targetIn.valid.poke(false.B)

      dut.io.traceIn.bits.gid.poke(7.U)
      dut.io.traceIn.bits.frameId.poke(terminalFrame.U)
      dut.io.traceIn.bits.pixel.poke(3.U)
      dut.io.traceIn.bits.blendOrder.poke(0.U)
      dut.io.traceIn.bits.coverage.poke(fpHalf.U)
      dut.io.traceIn.bits.transmittance.poke(fpHalf.U)
      (0 until 3).foreach { channel => dut.io.traceIn.bits.color(channel).poke(fpHalf.U) }
      driveProjection(dut.io.traceIn.bits.projection, gid = 7, frame = terminalFrame.toInt)
      dut.io.traceIn.bits.path.poke(ScdGsConsts.PathLight.U)
      dut.io.traceIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.traceIn.valid.poke(false.B)

      dut.io.invalidPairCount.expect(1.U)
      dut.io.stateWrite.valid.expect(false.B)
      dut.io.updateOut.valid.expect(false.B)
      dut.io.targetIn.ready.expect(false.B)
    }
  }
}
