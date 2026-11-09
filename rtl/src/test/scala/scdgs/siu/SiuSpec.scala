package scdgs.siu

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests projected impact, five-observation confirmation, and structural protection. */
class SiuSpec extends AnyFunSuite with Matchers with ChiselSim {
  test("SIU shared 3x3 FMA array preserves an identity projection contraction") {
    simulate(new SharedFma3x3) { dut =>
      for (row <- 0 until 3) {
        for (column <- 0 until 3) {
          dut.io.matrix(row)(column).poke((if (row == column) fpOne else fpZero).U)
        }
      }
      dut.io.vector(0).poke(fpTenth.U)
      dut.io.vector(1).poke(fpHalf.U)
      dut.io.vector(2).poke(fpZero.U)
      dut.clock.step()
      dut.io.out(0).expect(fpTenth.U)
      dut.io.out(1).expect(fpHalf.U)
      dut.io.out(2).expect(fpZero.U)
    }
  }

  test("SIU protects warmup, confirms low impact after five observations, and commits structural updates") {
    simulate(new SplatImpactUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      dut.io.propagationOut.ready.poke(true.B)
      dut.clock.step()

      for (frame <- 1 to 5) {
        driveUpdate(dut.io.updateIn.bits, gid = 7, frame = frame)
        driveProjection(dut.io.priorSummaryIn.bits, gid = 7, frame = frame - 1)
        dut.io.updateIn.valid.poke(true.B)
        dut.io.priorSummaryIn.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.updateIn.valid.poke(false.B)
        dut.io.priorSummaryIn.valid.poke(false.B)
        stepUntil(dut.clock, 160)(dut.io.propagationOut.valid.peek().litToBoolean)
        if (frame < 5) {
          dut.io.propagationOut.bits.propagation.expect(ScdGsConsts.PropagationCommit.U)
          dut.io.propagationOut.bits.effectiveStrength.expect(fpTenth.U)
        } else {
          dut.io.propagationOut.bits.propagation.expect(ScdGsConsts.PropagationConfirmation.U)
          dut.io.propagationOut.bits.effectiveStrength.expect(fpZero.U)
        }
        dut.clock.step()
      }

      driveUpdate(dut.io.updateIn.bits, gid = 7, frame = 6, structural = ScdGsConsts.StructuralInsert)
      driveProjection(dut.io.priorSummaryIn.bits, gid = 7, frame = 5)
      dut.io.updateIn.valid.poke(true.B)
      dut.io.priorSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      stepUntil(dut.clock, 160)(dut.io.propagationOut.valid.peek().litToBoolean)
      dut.io.propagationOut.bits.propagation.expect(ScdGsConsts.PropagationCommit.U)
      dut.io.propagationOut.bits.effectiveStrength.expect(fpTenth.U)
    }
  }

  test("SIU consumes but does not accept a malformed update and projection pair") {
    simulate(new SplatImpactUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      dut.io.propagationOut.ready.poke(true.B)
      dut.clock.step()

      driveUpdate(dut.io.updateIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorSummaryIn.bits, gid = 8, frame = 0)
      dut.io.updateIn.valid.poke(true.B)
      dut.io.priorSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
      dut.io.acceptedInput.expect(false.B)
      dut.clock.step()
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)

      dut.io.invalidInputCount.expect(1.U)
      dut.io.propagationOut.valid.expect(false.B)
    }
  }

  test("SIU rejects a nonpositive or non-normal projected radius") {
    simulate(new SplatImpactUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      dut.io.propagationOut.ready.poke(true.B)
      dut.clock.step()

      driveUpdate(dut.io.updateIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorSummaryIn.bits, gid = 7, frame = 0)
      dut.io.priorSummaryIn.bits.radius.poke(fpZero.U)
      dut.io.updateIn.valid.poke(true.B)
      dut.io.priorSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
      dut.io.acceptedInput.expect(false.B)
      dut.clock.step()
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)

      dut.io.invalidInputCount.expect(1.U)
      dut.io.propagationOut.valid.expect(false.B)
    }
  }

  test("SIU rejects a user projection as its prior training summary") {
    simulate(new SplatImpactUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      dut.io.propagationOut.ready.poke(true.B)
      dut.clock.step()

      driveUpdate(dut.io.updateIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorSummaryIn.bits, gid = 7, frame = 0, request = ScdGsConsts.RequestUser)
      dut.io.updateIn.valid.poke(true.B)
      dut.io.priorSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
      dut.io.acceptedInput.expect(false.B)
      dut.clock.step()
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)

      dut.io.invalidInputCount.expect(1.U)
      dut.io.propagationOut.valid.expect(false.B)
    }
  }

  test("SIU rejects an undefined structural marker rather than treating it as a commit") {
    simulate(new SplatImpactUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      dut.io.propagationOut.ready.poke(true.B)
      dut.clock.step()

      driveUpdate(dut.io.updateIn.bits, gid = 7, frame = 1, structural = 7)
      driveProjection(dut.io.priorSummaryIn.bits, gid = 7, frame = 0)
      dut.io.updateIn.valid.poke(true.B)
      dut.io.priorSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
      dut.io.acceptedInput.expect(false.B)
      dut.clock.step()
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)

      dut.io.invalidInputCount.expect(1.U)
      dut.io.propagationOut.valid.expect(false.B)
    }
  }

  test("SIU rejects a summary from the update frame rather than a preceding render") {
    simulate(new SplatImpactUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)
      dut.io.propagationOut.ready.poke(true.B)
      dut.clock.step()

      driveUpdate(dut.io.updateIn.bits, gid = 7, frame = 1)
      driveProjection(dut.io.priorSummaryIn.bits, gid = 7, frame = 1)
      dut.io.updateIn.valid.poke(true.B)
      dut.io.priorSummaryIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.updateIn.ready.peek().litToBoolean)
      dut.io.acceptedInput.expect(false.B)
      dut.clock.step()
      dut.io.updateIn.valid.poke(false.B)
      dut.io.priorSummaryIn.valid.poke(false.B)

      dut.io.invalidInputCount.expect(1.U)
      dut.io.propagationOut.valid.expect(false.B)
    }
  }
}
