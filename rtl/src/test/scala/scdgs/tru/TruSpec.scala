package scdgs.tru

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests frame-drained tier merge, transmittance stop, and training trajectory emission. */
class TruSpec extends AnyFunSuite with Matchers with ChiselSim {
  private def fp(hex: String): BigInt = BigInt(hex, 16)
  private def testParams(renderQueueDepth: Int): ScdGsParams =
    ScdGsParams(maxGaussians = 16, tileWidth = 2, renderQueueDepth = renderQueueDepth)

  test("TRU stops a late resident tile and globally orders the next frame") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 4))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(false.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      def enqueue(gid: Int, tier: Int, depth: BigInt): Unit = {
        driveIssue(dut.io.issueIn.bits, gid = gid, frame = 0, tier = tier, path = ScdGsConsts.PathFull, request = ScdGsConsts.RequestTraining, alpha = fpOne)
        dut.io.issueIn.bits.view.depth.poke(depth.U)
        dut.io.issueIn.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.issueIn.valid.poke(false.B)
      }

      // Coverage alpha is derived from state opacity, rather than copied from
      // an issue descriptor. Three early splats therefore establish T < 0.05
      // before the resident record is considered.
      enqueue(7, ScdGsConsts.TierActive, fpTenth)
      enqueue(9, ScdGsConsts.TierActive, BigInt("3e4ccccd", 16))
      enqueue(10, ScdGsConsts.TierActive, BigInt("3e99999a", 16))
      enqueue(8, ScdGsConsts.TierResident, BigInt("3ecccccd", 16))
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(7.U)
      dut.io.traceOut.bits.path.expect(ScdGsConsts.PathFull.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(9.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(10.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.earlyStopCount.peek().litValue == 1)
      dut.io.earlyStopCount.expect(1.U)

      // Let the empty first frame release drain mode. A new frame ID resets
      // per-tile transmittance while retaining the same hardware instance.
      dut.clock.step()

      def enqueueOrdered(gid: Int, tier: Int, tile: Int, depth: BigInt): Unit = {
        driveIssue(dut.io.issueIn.bits, gid = gid, frame = 1, tier = tier, path = ScdGsConsts.PathLight, request = ScdGsConsts.RequestTraining, alpha = fpHalf)
        dut.io.issueIn.bits.view.tile.poke(tile.U)
        dut.io.issueIn.bits.view.depth.poke(depth.U)
        dut.io.issueIn.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.issueIn.valid.poke(false.B)
      }

      enqueueOrdered(7, ScdGsConsts.TierActive, tile = 2, depth = fpTenth)
      enqueueOrdered(8, ScdGsConsts.TierResident, tile = 1, depth = fp98)
      enqueueOrdered(9, ScdGsConsts.TierCandidate, tile = 1, depth = BigInt("3e4ccccd", 16))
      enqueueOrdered(6, ScdGsConsts.TierCandidate, tile = 1, depth = BigInt("3e4ccccd", 16))
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)

      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(6.U)
      dut.io.traceOut.bits.blendOrder.expect(0.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(9.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(8.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(7.U)
    }
  }

  test("TRU evaluates all degree-3 SH bases and rounds light color through FP16") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 2))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      def enqueue(gid: Int, path: Int, depth: BigInt)(configure: => Unit): Unit = {
        driveIssue(
          dut.io.issueIn.bits,
          gid = gid,
          frame = 0,
          tier = ScdGsConsts.TierActive,
          path = path,
          request = ScdGsConsts.RequestTraining
        )
        dut.io.issueIn.bits.view.tile.poke(1.U)
        dut.io.issueIn.bits.view.depth.poke(depth.U)
        configure
        dut.io.issueIn.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.issueIn.valid.poke(false.B)
      }

      enqueue(gid = 5, path = ScdGsConsts.PathFull, depth = fpTenth) {
        val state = dut.io.issueIn.bits.residency.state
        val projection = dut.io.issueIn.bits.view
        (0 until state.sh.length).foreach { index => state.sh(index).poke(fpZero.U) }
        val coefficients = Seq("3d800000", "bd000000", "3d000000")
        for (channel <- 0 until 3; basis <- 0 until projection.shBasis.length) {
          state.sh(channel * projection.shBasis.length + basis).poke(fp(coefficients(channel)).U)
          projection.shBasis(basis).poke(fp("3e800000").U)
        }
      }
      enqueue(gid = 6, path = ScdGsConsts.PathLight, depth = fp98) {
        dut.io.issueIn.bits.colorEstimate(0).poke(fp("3f001000").U)
        dut.io.issueIn.bits.colorEstimate(1).poke(fp("3f001400").U)
        dut.io.issueIn.bits.colorEstimate(2).poke(fp("387fe000").U)
      }

      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)

      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(5.U)
      dut.io.traceOut.bits.path.expect(ScdGsConsts.PathFull.U)
      dut.io.traceOut.bits.pixel.expect(1.U)
      dut.io.traceOut.bits.color(0).expect(fp("3e800000").U)
      dut.io.traceOut.bits.color(1).expect(fp("be000000").U)
      dut.io.traceOut.bits.color(2).expect(fp("3e000000").U)
      dut.clock.step()

      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(6.U)
      dut.io.traceOut.bits.path.expect(ScdGsConsts.PathLight.U)
      dut.io.traceOut.bits.pixel.expect(1.U)
      dut.io.traceOut.bits.color(0).expect(fp("3f000000").U)
      dut.io.traceOut.bits.color(1).expect(fp("3f002000").U)
      dut.io.traceOut.bits.color(2).expect(fp("38800000").U)
      dut.clock.step()
      dut.io.shOpCount.expect(2.U)
    }
  }

  test("TRU permits independent trace and FRU render-read backpressure") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 1))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(false.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      driveIssue(
        dut.io.issueIn.bits,
        gid = 7,
        frame = 0,
        tier = ScdGsConsts.TierActive,
        path = ScdGsConsts.PathLight,
        request = ScdGsConsts.RequestTraining
      )
      dut.io.issueIn.valid.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)
      dut.clock.step()
      dut.io.issueIn.valid.poke(false.B)
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)

      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.renderReadOut.valid.expect(true.B)
      // The ready render-read link transfers while the trace remains held.
      dut.clock.step()
      dut.io.renderReadOut.valid.expect(false.B)

      dut.io.traceOut.ready.poke(true.B)
      dut.io.traceOut.valid.expect(true.B)
      dut.clock.step()
    }
  }

  test("TRU counts a held frameEnd as one empty frame boundary") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 1))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      driveIssue(
        dut.io.issueIn.bits,
        gid = 7,
        frame = 0,
        tier = ScdGsConsts.TierActive,
        path = ScdGsConsts.PathLight,
        request = ScdGsConsts.RequestTraining
      )
      dut.clock.step()

      dut.io.frameEnd.poke(true.B)
      dut.clock.step(3)
      dut.io.frameEnd.poke(false.B)

      // A held high level must not re-enter drain mode after the empty frame
      // has closed, so normal work remains immediately admissible.
      dut.io.issueIn.ready.expect(true.B)
    }
  }

  test("TRU discards an issue with an undefined path tag") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 1))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      driveIssue(
        dut.io.issueIn.bits,
        gid = 7,
        frame = 0,
        tier = ScdGsConsts.TierActive,
        path = 3,
        request = ScdGsConsts.RequestTraining
      )
      dut.io.issueIn.valid.poke(true.B)
      dut.io.issueIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.issueIn.valid.poke(false.B)
      dut.io.invalidInputCount.expect(1.U)

      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      dut.io.traceOut.valid.expect(false.B)
      dut.io.renderReadOut.valid.expect(false.B)
      dut.io.blendOpCount.expect(0.U)
    }
  }

  test("TRU rejects terminal-frame training work before it can create render side effects") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 1))) { dut =>
      val terminalFrame = (BigInt(1) << 16) - 1
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      driveIssue(
        dut.io.issueIn.bits,
        gid = 7,
        frame = terminalFrame.toInt,
        tier = ScdGsConsts.TierActive,
        path = ScdGsConsts.PathLight,
        request = ScdGsConsts.RequestTraining
      )
      dut.io.issueIn.valid.poke(true.B)
      dut.io.issueIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.issueIn.valid.poke(false.B)
      dut.io.invalidInputCount.expect(1.U)

      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      dut.io.traceOut.valid.expect(false.B)
      dut.io.renderReadOut.valid.expect(false.B)
      dut.io.blendOpCount.expect(0.U)
      dut.io.shOpCount.expect(0.U)
    }
  }

  test("TRU rejects a non-finite depth before global compositing") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 1))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      driveIssue(
        dut.io.issueIn.bits,
        gid = 7,
        frame = 0,
        tier = ScdGsConsts.TierActive,
        path = ScdGsConsts.PathLight,
        request = ScdGsConsts.RequestTraining
      )
      dut.io.issueIn.bits.view.depth.poke("h7fc00000".U)
      dut.io.issueIn.valid.poke(true.B)
      dut.io.issueIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.issueIn.valid.poke(false.B)
      dut.io.invalidInputCount.expect(1.U)

      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      dut.io.traceOut.valid.expect(false.B)
      dut.io.renderReadOut.valid.expect(false.B)
      dut.io.blendOpCount.expect(0.U)
    }
  }

  test("TRU uses numerical binary32 depth equality before the Gaussian-ID tie break") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 2))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      def enqueue(gid: Int, depth: BigInt): Unit = {
        driveIssue(
          dut.io.issueIn.bits,
          gid = gid,
          frame = 0,
          tier = ScdGsConsts.TierActive,
          path = ScdGsConsts.PathLight,
          request = ScdGsConsts.RequestTraining
        )
        dut.io.issueIn.bits.view.tile.poke(1.U)
        dut.io.issueIn.bits.view.depth.poke(depth.U)
        dut.io.issueIn.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.issueIn.valid.poke(false.B)
      }

      // Queue the larger GID first. +0.0 and -0.0 have distinct encodings but
      // equal IEEE-754 values, so gid=7 must still be selected first.
      enqueue(gid = 9, depth = fpZero)
      enqueue(gid = 7, depth = BigInt("80000000", 16))
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)

      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(7.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(9.U)
    }
  }

  test("TRU does not mix issue records from different frames into one closed batch") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 1))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      def drive(frame: Int, gid: Int): Unit = {
        driveIssue(
          dut.io.issueIn.bits,
          gid = gid,
          frame = frame,
          tier = ScdGsConsts.TierActive,
          path = ScdGsConsts.PathLight,
          request = ScdGsConsts.RequestTraining
        )
      }

      drive(frame = 0, gid = 7)
      dut.io.issueIn.valid.poke(true.B)
      dut.io.issueIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.issueIn.valid.poke(false.B)

      // The frame-1 record cannot share frame-0's frameEnd or blend order.
      drive(frame = 1, gid = 8)
      dut.io.issueIn.valid.poke(true.B)
      dut.io.issueIn.ready.expect(false.B)
      dut.io.issueIn.valid.poke(false.B)

      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(7.U)
      dut.io.traceOut.bits.frameId.expect(0.U)
      dut.clock.step()
      stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)

      drive(frame = 1, gid = 8)
      dut.io.issueIn.valid.poke(true.B)
      dut.io.issueIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.issueIn.valid.poke(false.B)
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(8.U)
      dut.io.traceOut.bits.frameId.expect(1.U)
    }
  }

  test("TRU does not reset a closed frame's blend order on a later empty boundary") {
    simulate(new TieredRenderUnit(testParams(renderQueueDepth = 2))) { dut =>
      dut.io.issueIn.valid.poke(false.B)
      dut.io.traceOut.ready.poke(true.B)
      dut.io.renderReadOut.ready.poke(true.B)
      dut.io.frameEnd.poke(false.B)
      dut.clock.step()

      def enqueue(gid: Int, depth: BigInt): Unit = {
        driveIssue(
          dut.io.issueIn.bits,
          gid = gid,
          frame = 0,
          tier = ScdGsConsts.TierActive,
          path = ScdGsConsts.PathLight,
          request = ScdGsConsts.RequestTraining
        )
        dut.io.issueIn.bits.view.depth.poke(depth.U)
        dut.io.issueIn.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.issueIn.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.issueIn.valid.poke(false.B)
      }

      enqueue(gid = 7, depth = fpTenth)
      enqueue(gid = 8, depth = fp98)
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)

      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(7.U)
      dut.io.traceOut.bits.blendOrder.expect(0.U)
      dut.clock.step()

      // A later frame edge can arrive while this closed frame is backpressured.
      // It represents an empty batch because `issueIn` is blocked by draining.
      dut.io.traceOut.ready.poke(false.B)
      dut.clock.step()
      dut.io.frameEnd.poke(true.B)
      dut.clock.step()
      dut.io.frameEnd.poke(false.B)

      dut.io.traceOut.ready.poke(true.B)
      stepUntil(dut.clock, 4)(dut.io.traceOut.valid.peek().litToBoolean)
      dut.io.traceOut.bits.gid.expect(8.U)
      dut.io.traceOut.bits.blendOrder.expect(1.U)
    }
  }
}
