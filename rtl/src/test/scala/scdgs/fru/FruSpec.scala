package scdgs.fru

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests commit/read event fusion, h/d saturation, allocation, and aging. */
class FruSpec extends AnyFunSuite with Matchers with ChiselSim {
  test("FRU fuses events, ranks active/resident entries, and ages an untouched window") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 16, activeCapacity = 1, residentCapacity = 1))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.valid.poke(false.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      drivePropagation(dut.io.commitIn.bits, gid = 7, frame = 0, strength = fpOne)
      dut.io.commitIn.valid.poke(true.B)
      dut.io.residencyOut.bits.sampleCount.expect(1.U)
      dut.io.residencyOut.bits.energy.expect(fpOne.U)
      dut.io.metadataSnapshot.valid.expect(true.B)
      dut.io.metadataSnapshotGid.expect(7.U)
      dut.io.metadataSnapshot.bits.tier.expect(ScdGsConsts.TierCandidate.U)
      dut.io.metadataSnapshot.bits.row.expect(7.U)
      dut.io.metadataSnapshot.bits.hotness.expect(1.U)
      dut.io.metadataSnapshot.bits.delta.expect(255.U)
      dut.io.metadataSnapshot.bits.energy.expect(fpOne.U)
      dut.io.metadataSnapshot.bits.samples.expect(1.U)
      dut.io.metadataSnapshot.bits.touched.expect(true.B)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)

      // A confirmation remains a pure FRU metadata event: it updates the
      // post-event snapshot but must not create a StateSPM residency transfer.
      drivePropagation(dut.io.commitIn.bits, gid = 7, frame = 0, strength = fpZero)
      dut.io.commitIn.bits.propagation.poke(ScdGsConsts.PropagationConfirmation.U)
      dut.io.commitIn.valid.poke(true.B)
      dut.io.residencyOut.valid.expect(false.B)
      dut.io.metadataSnapshot.valid.expect(true.B)
      dut.io.metadataSnapshotGid.expect(7.U)
      dut.io.metadataSnapshot.bits.tier.expect(ScdGsConsts.TierCandidate.U)
      dut.io.metadataSnapshot.bits.row.expect(7.U)
      dut.io.metadataSnapshot.bits.hotness.expect(1.U)
      dut.io.metadataSnapshot.bits.delta.expect(255.U)
      dut.io.metadataSnapshot.bits.energy.expect(fpOne.U)
      dut.io.metadataSnapshot.bits.samples.expect(2.U)
      dut.io.metadataSnapshot.bits.touched.expect(true.B)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)

      for (read <- 0 until 15) {
        dut.io.renderReadIn.bits.gid.poke(7.U)
        dut.io.renderReadIn.bits.frameId.poke(0.U)
        dut.io.renderReadIn.valid.poke(true.B)
        dut.io.metadataSnapshot.valid.expect(true.B)
        dut.io.metadataSnapshotGid.expect(7.U)
        dut.io.metadataSnapshot.bits.hotness.expect(math.min(15, read + 2).U)
        dut.io.metadataSnapshot.bits.delta.expect(255.U)
        dut.io.metadataSnapshot.bits.energy.expect(fpOne.U)
        dut.io.metadataSnapshot.bits.samples.expect(2.U)
        dut.io.metadataSnapshot.bits.touched.expect(true.B)
        dut.clock.step()
      }
      dut.io.renderReadIn.valid.poke(false.B)

      drivePropagation(dut.io.commitIn.bits, gid = 8, frame = 0, strength = fpHalf)
      dut.io.commitIn.valid.poke(true.B)
      dut.io.metadataSnapshot.valid.expect(true.B)
      dut.io.metadataSnapshotGid.expect(8.U)
      // FRU uses truncation toward zero for the 8-bit recent-strength field:
      // floor(0.5 * 255) is 127.
      dut.io.metadataSnapshot.bits.delta.expect(127.U)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)
      for (_ <- 0 until 5) {
        dut.io.frameEnd.poke(true.B)
        dut.clock.step()
        dut.io.frameEnd.poke(false.B)
        dut.clock.step()
      }
      // Holding a new commit while allocation owns the directory cannot leak a
      // speculative residency record to StateSPM or CIU.
      drivePropagation(dut.io.commitIn.bits, gid = 9, frame = 1, strength = fpTenth)
      dut.io.commitIn.valid.poke(true.B)
      dut.io.commitIn.ready.expect(false.B)
      dut.io.residencyOut.valid.expect(false.B)
      dut.io.commitIn.valid.poke(false.B)
      stepUntil(dut.clock, 128)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 7 &&
          dut.io.metadataSnapshot.bits.tier.peek().litValue == BigInt(ScdGsConsts.TierActive) &&
          dut.io.metadataSnapshot.bits.delta.peek().litValue == 0
      )
      dut.io.metadataSnapshot.bits.tier.expect(ScdGsConsts.TierActive.U)
      dut.io.metadataSnapshot.bits.delta.expect(0.U)
      // The observed GID clears early in the sequential aging pass. Let the
      // pass retire every remaining entry before starting the next window.
      dut.clock.step(16)

      for (_ <- 0 until 5) {
        dut.io.frameEnd.poke(true.B)
        dut.clock.step()
        dut.io.frameEnd.poke(false.B)
        dut.clock.step()
      }
      stepUntil(dut.clock, 128)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 7 &&
          dut.io.metadataSnapshot.bits.hotness.peek().litValue == 7 &&
          dut.io.metadataSnapshot.bits.delta.peek().litValue == 0
      )
      dut.io.metadataSnapshot.bits.hotness.expect(7.U)
      dut.clock.step(16)
      dut.io.candidateMissIn.poke(true.B)
      dut.clock.step()
      dut.io.candidateMissIn.poke(false.B)
      dut.io.candidateRefillCount.peek().litValue should be > BigInt(0)

      // GID 15 is scanned last. Its h=8 exceeds the aged h=7 of GID 7, so a
      // terminal-cycle omission would incorrectly retain GID 7 as active.
      drivePropagation(dut.io.commitIn.bits, gid = 15, frame = 10, strength = fpOne)
      dut.io.commitIn.valid.poke(true.B)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)
      for (_ <- 0 until 7) {
        dut.io.renderReadIn.bits.gid.poke(15.U)
        dut.io.renderReadIn.bits.frameId.poke(10.U)
        dut.io.renderReadIn.valid.poke(true.B)
        dut.clock.step()
      }
      dut.io.renderReadIn.valid.poke(false.B)

      for (_ <- 0 until 5) {
        dut.io.frameEnd.poke(true.B)
        dut.clock.step()
        dut.io.frameEnd.poke(false.B)
        dut.clock.step()
      }
      stepUntil(dut.clock, 128)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 7 &&
          dut.io.metadataSnapshot.bits.tier.peek().litValue == BigInt(ScdGsConsts.TierResident)
      )
      dut.io.metadataSnapshot.bits.tier.expect(ScdGsConsts.TierResident.U)
      stepUntil(dut.clock, 128)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 15 &&
          dut.io.metadataSnapshot.bits.tier.peek().litValue == BigInt(ScdGsConsts.TierActive)
      )
      dut.io.metadataSnapshot.bits.tier.expect(ScdGsConsts.TierActive.U)
    }
  }

  test("FRU assigns a gid-addressed candidate row to an initial render read") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.bits.gid.poke(13.U)
      dut.io.renderReadIn.bits.frameId.poke(0.U)
      dut.io.renderReadIn.valid.poke(true.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)

      dut.io.renderReadIn.ready.expect(true.B)
      dut.io.metadataSnapshot.valid.expect(true.B)
      dut.io.metadataSnapshotGid.expect(13.U)
      dut.io.metadataSnapshot.bits.tier.expect(ScdGsConsts.TierCandidate.U)
      dut.io.metadataSnapshot.bits.row.expect(13.U)
      dut.io.metadataSnapshot.bits.hotness.expect(1.U)
      dut.clock.step()

      dut.io.renderReadIn.valid.poke(false.B)
    }
  }

  test("FRU consumes an invalid commit without emitting a residency record") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.valid.poke(false.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(false.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      drivePropagation(dut.io.commitIn.bits, gid = 31, frame = 0, strength = fpTenth)
      dut.io.commitIn.valid.poke(true.B)
      dut.io.commitIn.ready.expect(true.B)
      dut.io.residencyOut.valid.expect(false.B)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)

      dut.io.invalidInputCount.expect(1.U)
    }
  }

  test("FRU counts a held frameEnd as one boundary") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 4, activeCapacity = 1, residentCapacity = 1))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.valid.poke(false.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      drivePropagation(dut.io.commitIn.bits, gid = 2, frame = 0, strength = fpTenth)
      dut.io.commitIn.valid.poke(true.B)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)

      dut.io.frameEnd.poke(true.B)
      dut.clock.step(2)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()
      for (_ <- 0 until 3) {
        dut.io.frameEnd.poke(true.B)
        dut.clock.step()
        dut.io.frameEnd.poke(false.B)
        dut.clock.step()
      }

      // Four rising edges have occurred, so the fifth-frame allocator must
      // still be idle. A level-sensitive implementation would have started it.
      dut.io.commitIn.ready.expect(true.B)
    }
  }

  test("FRU forms a correctly rounded three-sample effective-strength mean") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 16))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.valid.poke(false.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      val quarter = BigInt("3e800000", 16)
      Seq(fpHalf, quarter, quarter).zipWithIndex.foreach { case (strength, sample) =>
        drivePropagation(dut.io.commitIn.bits, gid = 7, frame = 0, strength = strength)
        dut.io.commitIn.valid.poke(true.B)
        dut.io.commitIn.ready.expect(true.B)
        dut.io.residencyOut.valid.expect(true.B)
        if (sample == 2) {
          dut.io.residencyOut.bits.energy.expect(BigInt("3eaaaaab", 16).U)
          dut.io.residencyOut.bits.sampleCount.expect(3.U)
          dut.io.metadataSnapshot.bits.energy.expect(fpOne.U)
          dut.io.metadataSnapshot.bits.samples.expect(3.U)
        }
        dut.clock.step()
        dut.io.commitIn.valid.poke(false.B)
      }
    }
  }

  test("FRU publishes the final allocator and aging metadata snapshot") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 4, activeCapacity = 1, residentCapacity = 1))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.valid.poke(false.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      drivePropagation(dut.io.commitIn.bits, gid = 2, frame = 0, strength = fpOne)
      dut.io.commitIn.valid.poke(true.B)
      dut.clock.step()
      dut.io.commitIn.valid.poke(false.B)

      for (_ <- 0 until 5) {
        dut.io.frameEnd.poke(true.B)
        dut.clock.step()
        dut.io.frameEnd.poke(false.B)
        dut.clock.step()
      }

      stepUntil(dut.clock, 96)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 2 &&
          dut.io.metadataSnapshot.bits.tier.peek().litValue == BigInt(ScdGsConsts.TierActive) &&
          dut.io.metadataSnapshot.bits.delta.peek().litValue == 0 &&
          dut.io.metadataSnapshot.bits.samples.peek().litValue == 0 &&
          !dut.io.metadataSnapshot.bits.touched.peek().litToBoolean
      )
      dut.io.metadataSnapshot.bits.row.expect(0.U)
      dut.io.metadataSnapshot.bits.hotness.expect(1.U)
      dut.io.metadataSnapshot.bits.energy.expect(fpZero.U)
    }
  }

  test("FRU breaks equal h and d priority ties with ascending Gaussian ID") {
    simulate(new FusedResidencyUnit(ScdGsParams(maxGaussians = 4, activeCapacity = 1, residentCapacity = 1))) { dut =>
      dut.io.commitIn.valid.poke(false.B)
      dut.io.renderReadIn.valid.poke(false.B)
      dut.io.candidateMissIn.poke(false.B)
      dut.io.residencyOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      def commit(gid: Int): Unit = {
        drivePropagation(dut.io.commitIn.bits, gid = gid, frame = 0, strength = fpHalf)
        dut.io.commitIn.valid.poke(true.B)
        dut.io.commitIn.ready.expect(true.B)
        dut.clock.step()
        dut.io.commitIn.valid.poke(false.B)
      }

      // Queue the larger identifier first. Both entries have h=1 and d=127,
      // so only the paper-defined ascending-GID final key can select gid=2.
      commit(3)
      commit(2)
      for (_ <- 0 until 5) {
        dut.io.frameEnd.poke(true.B)
        dut.clock.step()
        dut.io.frameEnd.poke(false.B)
        dut.clock.step()
      }

      stepUntil(dut.clock, 96)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 2 &&
          dut.io.metadataSnapshot.bits.tier.peek().litValue == BigInt(ScdGsConsts.TierActive)
      )
      dut.io.metadataSnapshot.bits.row.expect(0.U)
      stepUntil(dut.clock, 96)(
        dut.io.metadataSnapshot.valid.peek().litToBoolean &&
          dut.io.metadataSnapshotGid.peek().litValue == 3 &&
          dut.io.metadataSnapshot.bits.tier.peek().litValue == BigInt(ScdGsConsts.TierResident)
      )
      dut.io.metadataSnapshot.bits.row.expect(0.U)
    }
  }
}
