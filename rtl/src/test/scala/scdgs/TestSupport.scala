package scdgs

import chisel3._
import chisel3.simulator._
import chisel3.simulator.HasSimulator
import chisel3.simulator.HasSimulator.simulators.verilator
import scdgs.bundle._
import scdgs.config._
import svsim.CommonCompilationSettings
import svsim.CommonCompilationSettings.AvailableParallelism.UpTo
import svsim.CommonCompilationSettings.OptimizationStyle.OptimizeForCompilationSpeed
import svsim.verilator.Backend.CompilationSettings.Parallelism.Uniform

/** Shared full-record drivers. Tests intentionally never recreate a scalar-only update path. */
object TestSupport extends PeekPokeAPI {
  private val defaultTestParallelism = 4
  private val testParallelism = sys.props.get("scdgs.testParallelism") match {
    case None => defaultTestParallelism
    case Some(raw) =>
      raw.toIntOption.filter(_ > 0).getOrElse(
        throw new IllegalArgumentException("scdgs.testParallelism must be a positive integer")
      )
  }

  // Full-order binary32 networks need bounded C++ units, but excessively fine
  // splitting turns one top-level test into thousands of serial compiler
  // invocations. The default keeps the simulator portable, while the explicit
  // property above lets a well-provisioned host shorten only test compilation.
  implicit val regressionSimulator: HasSimulator = verilator(
    compilationSettings = CommonCompilationSettings(
      optimizationStyle = OptimizeForCompilationSpeed,
      availableParallelism = UpTo(testParallelism)
    ),
    verilatorSettings = svsim.verilator.Backend.CompilationSettings.default
      .withOutputSplit(Some(1000))
      .withOutputSplitCFuncs(Some(200))
      .withParallelism(Some(Uniform.default.withNum(testParallelism)))
  )

  val fpZero = BigInt("00000000", 16)
  val fpOne = BigInt("3f800000", 16)
  val fpHalf = BigInt("3f000000", 16)
  val fpTenth = BigInt("3dcccccd", 16)
  val fpHundredth = BigInt("3c23d70a", 16)
  val fp98 = BigInt("3f7ae148", 16)

  def stepUntil(clock: Clock, maximum: Int)(condition: => Boolean): Unit = {
    var steps = 0
    while (!condition && steps < maximum) {
      clock.step()
      steps += 1
    }
    assert(condition, s"condition was not met within $maximum cycles")
  }

  def driveState(bundle: GaussianStateBundle, gid: Int, opacity: BigInt = fpHalf): Unit = {
    bundle.gid.poke(gid.U)
    bundle.mean(0).poke(fpTenth.U)
    bundle.mean(1).poke(fpTenth.U)
    bundle.mean(2).poke(fpTenth.U)
    (0 until 6).foreach { index =>
      bundle.covariance(index).poke((if (index == 0 || index == 3 || index == 5) fpTenth else fpZero).U)
    }
    bundle.opacity.poke(opacity.U)
    (0 until bundle.sh.length).foreach { index =>
      bundle.sh(index).poke((if (index % 16 == 0) fpHalf else fpTenth).U)
    }
  }

  def driveUpdate(
      bundle: CanonicalUpdateBundle,
      gid: Int,
      frame: Int,
      strength: BigInt = fpTenth,
      structural: Int = ScdGsConsts.StructuralNone
  ): Unit = {
    bundle.gid.poke(gid.U)
    bundle.frameId.poke(frame.U)
    bundle.deltaMu(0).poke(fpTenth.U)
    bundle.deltaMu(1).poke(fpTenth.U)
    bundle.deltaMu(2).poke(fpZero.U)
    (0 until 6).foreach { index => bundle.deltaSigma(index).poke((if (index == 0 || index == 3 || index == 5) fpTenth else fpZero).U) }
    bundle.deltaOpacity.poke(fpTenth.U)
    (0 until bundle.deltaSh.length).foreach { index => bundle.deltaSh(index).poke((if (index % 16 == 0) fpTenth else fpZero).U) }
    bundle.strength.poke(strength.U)
    bundle.structural.poke(structural.U)
  }

  def driveProjection(
      bundle: ProjectionSummaryBundle,
      gid: Int,
      frame: Int,
      request: Int = ScdGsConsts.RequestTraining,
      area: BigInt = fpHundredth,
      tile: Int = 3,
      depth: BigInt = fpTenth
  ): Unit = {
    bundle.gid.poke(gid.U)
    bundle.frameId.poke(frame.U)
    for (row <- 0 until 3; column <- 0 until 3) {
      bundle.rCam(row)(column).poke((if (row == column) fpOne else fpZero).U)
    }
    for (row <- 0 until 2; column <- 0 until 3) {
      bundle.jPi(row)(column).poke((if (row == column) fpOne else fpZero).U)
      bundle.jCov(row)(column).poke((if (row == column) fpOne else fpZero).U)
    }
    bundle.viewDirection(0).poke(fpOne.U)
    bundle.viewDirection(1).poke(fpZero.U)
    bundle.viewDirection(2).poke(fpZero.U)
    (0 until bundle.shBasis.length).foreach { basis => bundle.shBasis(basis).poke((if (basis == 0) fpOne else fpZero).U) }
    bundle.center(0).poke(fpZero.U)
    bundle.center(1).poke(fpZero.U)
    bundle.conic(0).poke(fpOne.U)
    bundle.conic(1).poke(fpZero.U)
    bundle.conic(2).poke(fpOne.U)
    bundle.radius.poke(fpOne.U)
    bundle.area.poke(area.U)
    bundle.opacity.poke(fpHalf.U)
    (0 until 3).foreach { channel => bundle.viewColor(channel).poke(fpTenth.U) }
    bundle.requestKind.poke(request.U)
    bundle.tile.poke(tile.U)
    bundle.depth.poke(depth.U)
  }

  def drivePropagation(
      bundle: PropagationBundle,
      gid: Int,
      frame: Int,
      strength: BigInt = fpTenth,
      structural: Int = ScdGsConsts.StructuralNone,
      request: Int = ScdGsConsts.RequestTraining,
      area: BigInt = fpHundredth
  ): Unit = {
    driveUpdate(bundle.update, gid, frame, strength, structural)
    driveProjection(bundle.projection, gid, frame, request, area)
    bundle.psi.poke(fpTenth.U)
    bundle.phi.poke(fpTenth.U)
    bundle.effectiveStrength.poke(strength.U)
    bundle.propagation.poke(ScdGsConsts.PropagationCommit.U)
  }

  def driveResidency(
      bundle: ResidencyBundle,
      gid: Int,
      frame: Int,
      tier: Int = ScdGsConsts.TierCandidate,
      request: Int = ScdGsConsts.RequestTraining,
      area: BigInt = fpHundredth,
      sampleCount: Int = 5,
      energy: BigInt = fpZero,
      structural: Int = ScdGsConsts.StructuralNone,
      alpha: BigInt = fpHalf,
      stateReady: Boolean = true
  ): Unit = {
    drivePropagation(bundle.propagation, gid, frame, structural = structural, request = request, area = area)
    driveState(bundle.state, gid, alpha)
    bundle.tier.poke(tier.U)
    bundle.row.poke(gid.U)
    bundle.candidateRefill.poke((tier == ScdGsConsts.TierCandidate).B)
    bundle.stateReady.poke(stateReady.B)
    bundle.energy.poke(energy.U)
    bundle.sampleCount.poke(sampleCount.U)
  }

  def driveIssue(bundle: IssueBundle, gid: Int, frame: Int, tier: Int, path: Int, request: Int, alpha: BigInt = fpHalf): Unit = {
    driveResidency(bundle.residency, gid, frame, tier, ScdGsConsts.RequestTraining, area = fpOne, energy = fpOne, alpha = alpha)
    driveProjection(bundle.view, gid, frame, request, area = fpOne)
    bundle.path.poke(path.U)
    bundle.supportArea.poke(fpOne.U)
    (0 until 3).foreach { channel => bundle.colorEstimate(channel).poke(fpTenth.U) }
    bundle.renderingPotential.poke(fpTenth.U)
    bundle.kappa.poke(fpZero.U)
    bundle.rho.poke(fpOne.U)
    bundle.omega.poke(fpOne.U)
  }

  def driveOptimizerState(bundle: OptimizerStateBundle, gid: Int): Unit = {
    bundle.gid.poke(gid.U)
    driveState(bundle.state, gid)
    (0 until bundle.adamMoment.length).foreach { index =>
      bundle.adamMoment(index).poke(fpZero.U)
      bundle.adamVariance(index).poke(fpZero.U)
    }
  }

  def driveStateWrite(bundle: StateWriteBundle, gid: Int): Unit = {
    bundle.gid.poke(gid.U)
    driveState(bundle.state, gid)
    (0 until bundle.adamMoment.length).foreach { index =>
      bundle.adamMoment(index).poke(fpZero.U)
      bundle.adamVariance(index).poke(fpZero.U)
    }
  }

  def driveCandidateFill(bundle: CandidateFillBundle, gid: Int): Unit = {
    driveState(bundle.state, gid)
    (0 until bundle.adamMoment.length).foreach { index =>
      bundle.adamMoment(index).poke(fpZero.U)
      bundle.adamVariance(index).poke(fpZero.U)
    }
  }
}
