package scdgs.ciu

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.config._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests the P/kappa/rho/omega path rule and training filter fallback. */
class CiuSpec extends AnyFunSuite with Matchers with ChiselSim {
  private def fp(hex: String): BigInt = BigInt(hex, 16)
  private val testParams = ScdGsParams(maxGaussians = 16)

  private def send(dut: ConditionedIssueUnit, request: Int, structural: Int = ScdGsConsts.StructuralNone): BigInt = {
    driveResidency(dut.io.residencyIn.bits, gid = 7, frame = 1, tier = ScdGsConsts.TierCandidate, request = ScdGsConsts.RequestTraining, area = fpHundredth, sampleCount = 5, energy = fpZero, structural = structural)
    driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = request, area = fpHundredth)
    dut.io.residencyIn.valid.poke(true.B)
    dut.io.currentViewIn.valid.poke(true.B)
    dut.io.issueOut.ready.poke(true.B)
    val path = dut.io.issueOut.bits.path.peek().litValue
    dut.clock.step()
    dut.io.residencyIn.valid.poke(false.B)
    dut.io.currentViewIn.valid.poke(false.B)
    path
  }

  test("CIU evaluates issue equations and keeps training work on light path") {
    simulate(new ConditionedIssueUnit(testParams)) { dut =>
      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.issueOut.ready.poke(true.B)
      dut.clock.step()
      send(dut, ScdGsConsts.RequestUser) shouldBe BigInt(ScdGsConsts.PathFilter)
      send(dut, ScdGsConsts.RequestTraining) shouldBe BigInt(ScdGsConsts.PathLight)
      send(dut, ScdGsConsts.RequestUser, ScdGsConsts.StructuralReset) shouldBe BigInt(ScdGsConsts.PathFull)
      dut.io.filterCount.expect(1.U)
      dut.io.lightCount.expect(1.U)
      dut.io.fullCount.expect(1.U)
    }
  }

  test("CIU forces incomplete noncandidate history to full while preserving the candidate exception") {
    simulate(new ConditionedIssueUnit(testParams)) { dut =>
      def pathFor(tier: Int): BigInt = {
        driveResidency(
          dut.io.residencyIn.bits,
          gid = 7,
          frame = 1,
          tier = tier,
          request = ScdGsConsts.RequestTraining,
          area = fpHundredth,
          sampleCount = 4,
          energy = fpZero
        )
        driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = ScdGsConsts.RequestUser, area = fpHundredth)
        dut.io.residencyIn.valid.poke(true.B)
        dut.io.currentViewIn.valid.poke(true.B)
        dut.io.issueOut.ready.poke(true.B)
        val path = dut.io.issueOut.bits.path.peek().litValue
        dut.clock.step()
        dut.io.residencyIn.valid.poke(false.B)
        dut.io.currentViewIn.valid.poke(false.B)
        path
      }

      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.issueOut.ready.poke(true.B)
      dut.clock.step()

      pathFor(ScdGsConsts.TierActive) shouldBe BigInt(ScdGsConsts.PathFull)
      pathFor(ScdGsConsts.TierCandidate) shouldBe BigInt(ScdGsConsts.PathFilter)
    }
  }

  test("CIU consumes a mismatched current view without issuing work") {
    simulate(new ConditionedIssueUnit(testParams)) { dut =>
      driveResidency(
        dut.io.residencyIn.bits,
        gid = 7,
        frame = 1,
        tier = ScdGsConsts.TierCandidate,
        request = ScdGsConsts.RequestTraining,
        sampleCount = 5,
        energy = fpZero
      )
      driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 2, request = ScdGsConsts.RequestTraining)
      dut.io.residencyIn.valid.poke(true.B)
      dut.io.currentViewIn.valid.poke(true.B)
      dut.io.issueOut.ready.poke(true.B)
      dut.io.issueOut.valid.expect(false.B)
      dut.io.residencyIn.ready.expect(true.B)
      dut.io.currentViewIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.invalidInputCount.expect(1.U)
      dut.io.fullCount.expect(0.U)
      dut.io.lightCount.expect(0.U)
      dut.io.filterCount.expect(0.U)
    }
  }

  test("CIU rejects an incomplete StateSPM response without issuing work") {
    simulate(new ConditionedIssueUnit(testParams)) { dut =>
      driveResidency(
        dut.io.residencyIn.bits,
        gid = 7,
        frame = 1,
        tier = ScdGsConsts.TierCandidate,
        request = ScdGsConsts.RequestTraining,
        sampleCount = 5,
        energy = fpZero,
        stateReady = false
      )
      driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = ScdGsConsts.RequestTraining)
      dut.io.residencyIn.valid.poke(true.B)
      dut.io.currentViewIn.valid.poke(true.B)
      dut.io.issueOut.ready.poke(true.B)
      dut.io.issueOut.valid.expect(false.B)
      dut.io.residencyIn.ready.expect(true.B)
      dut.io.currentViewIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.invalidInputCount.expect(1.U)
      dut.io.fullCount.expect(0.U)
      dut.io.lightCount.expect(0.U)
      dut.io.filterCount.expect(0.U)
    }
  }

  test("CIU saturates rendering potential when every contributing term reaches one") {
    simulate(new ConditionedIssueUnit(testParams)) { dut =>
      driveResidency(
        dut.io.residencyIn.bits,
        gid = 7,
        frame = 1,
        tier = ScdGsConsts.TierCandidate,
        request = ScdGsConsts.RequestTraining,
        sampleCount = 5,
        energy = fpZero,
        alpha = fp("40000000")
      )
      Seq(0, 3).foreach { index =>
        dut.io.residencyIn.bits.state.covariance(index).poke(fpOne.U)
      }
      (0 until 3).foreach { channel =>
        dut.io.residencyIn.bits.state.sh(channel * 16).poke(fp("40000000").U)
      }
      driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = ScdGsConsts.RequestTraining)
      dut.io.residencyIn.valid.poke(true.B)
      dut.io.currentViewIn.valid.poke(true.B)
      dut.io.issueOut.ready.poke(true.B)

      dut.io.issueOut.bits.supportArea.expect(fpOne.U)
      dut.io.issueOut.bits.colorEstimate.foreach(_.expect(fp("40000000").U))
      dut.io.issueOut.bits.renderingPotential.expect(fpOne.U)
      dut.clock.step()
      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
    }
  }

  test("CIU projects a non-diagonal covariance while retaining SIU coverage fields") {
    simulate(new ConditionedIssueUnit(testParams)) { dut =>
      def driveNonTrivialRecord(request: Int): Unit = {
        driveResidency(
          dut.io.residencyIn.bits,
          gid = 7,
          frame = 1,
          tier = ScdGsConsts.TierCandidate,
          request = ScdGsConsts.RequestTraining,
          sampleCount = 5,
          energy = fp("3e800000")
        )

        val covariance = Seq("3e800000", "3d800000", "bd000000", "3f000000", "3e000000", "3f400000")
        covariance.zipWithIndex.foreach { case (value, index) =>
          dut.io.residencyIn.bits.state.covariance(index).poke(fp(value).U)
        }
        val jacobian = Seq(
          Seq("3f000000", "be800000", "3e000000"),
          Seq("3e800000", "3f000000", "bf000000")
        )
        val basis = Seq("3f800000", "3f000000", "be800000", "3e000000")
        (0 until dut.io.residencyIn.bits.state.sh.length).foreach { index =>
          dut.io.residencyIn.bits.state.sh(index).poke(fpZero.U)
        }
        val lowOrderCoefficients = Seq(
          Seq("3f000000", "3e800000", "be800000", "3e000000"),
          Seq("3e800000", "3f000000", "3e000000", "be800000"),
          Seq("3f400000", "be800000", "3f000000", "3e800000")
        )
        for (channel <- 0 until 3; basisIndex <- 0 until 4) {
          dut.io.residencyIn.bits.state.sh(channel * 16 + basisIndex).poke(fp(lowOrderCoefficients(channel)(basisIndex)).U)
        }
        driveProjection(dut.io.currentViewIn.bits, gid = 7, frame = 1, request = request, area = fpHundredth)
        for (row <- 0 until 2; column <- 0 until 3) {
          dut.io.currentViewIn.bits.jCov(row)(column).poke(fp(jacobian(row)(column)).U)
        }
        (0 until dut.io.currentViewIn.bits.shBasis.length).foreach { index =>
          dut.io.currentViewIn.bits.shBasis(index).poke((if (index < basis.length) fp(basis(index)) else fpZero).U)
        }
        dut.io.residencyIn.valid.poke(true.B)
        dut.io.currentViewIn.valid.poke(true.B)
        dut.io.issueOut.ready.poke(true.B)
      }

      def expectEquationFields(path: Int): Unit = {
        dut.io.issueOut.bits.path.expect(path.U)
        dut.io.issueOut.bits.view.conic(0).expect(fpOne.U)
        dut.io.issueOut.bits.view.conic(1).expect(fpZero.U)
        dut.io.issueOut.bits.view.conic(2).expect(fpOne.U)
        dut.io.issueOut.bits.view.radius.expect(fpOne.U)
        dut.io.issueOut.bits.view.area.expect(fpHundredth.U)
        dut.io.issueOut.bits.supportArea.expect(fp("3e4417c0").U)
        dut.io.issueOut.bits.colorEstimate(0).expect(fp("3f340000").U)
        dut.io.issueOut.bits.colorEstimate(1).expect(fp("3ee00000").U)
        dut.io.issueOut.bits.colorEstimate(2).expect(fp("3f080000").U)
        dut.io.issueOut.bits.renderingPotential.expect(fp("3d8899e1").U)
        dut.io.issueOut.bits.kappa.expect(fp("3f400000").U)
        dut.io.issueOut.bits.rho.expect(fp("3e999cda").U)
        dut.io.issueOut.bits.omega.expect(fp("3d6b54ae").U)
      }

      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
      dut.io.issueOut.ready.poke(true.B)
      dut.clock.step()

      driveNonTrivialRecord(ScdGsConsts.RequestTraining)
      expectEquationFields(ScdGsConsts.PathLight)
      dut.clock.step()
      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)

      driveNonTrivialRecord(ScdGsConsts.RequestUser)
      expectEquationFields(ScdGsConsts.PathFilter)
      dut.clock.step()
      dut.io.residencyIn.valid.poke(false.B)
      dut.io.currentViewIn.valid.poke(false.B)
    }
  }
}
