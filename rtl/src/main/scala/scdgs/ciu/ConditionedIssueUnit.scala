package scdgs.ciu

import chisel3._
import chisel3.util.Decoupled
import scdgs.bundle._
import scdgs.config._
import scdgs.fp.Fp32

/** Stage III projection, support and issue-rule pipeline. */
class ConditionedIssueUnit(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val residencyIn = Flipped(Decoupled(new ResidencyBundle(p)))
    val currentViewIn = Flipped(Decoupled(new ProjectionSummaryBundle(p)))
    val issueOut = Decoupled(new IssueBundle(p))
    val fullCount = Output(UInt(32.W))
    val lightCount = Output(UInt(32.W))
    val filterCount = Output(UInt(32.W))
    val invalidInputCount = Output(UInt(32.W))
  })

  private def dot(left: Seq[UInt], right: Seq[UInt]): UInt = left.zip(right).map { case (a, b) => Fp32.mul(a, b) }.reduce(Fp32.add)
  val full = RegInit(0.U(32.W))
  val light = RegInit(0.U(32.W))
  val filter = RegInit(0.U(32.W))
  val invalid = RegInit(0.U(32.W))
  val summary = io.currentViewIn.bits
  val state = io.residencyIn.bits.state
  val matchingView =
    summary.gid === io.residencyIn.bits.propagation.update.gid &&
      summary.frameId === io.residencyIn.bits.propagation.update.frameId
  val pairedInput = io.residencyIn.valid && io.currentViewIn.valid
  val completeState = io.residencyIn.bits.stateReady
  val projection = Wire(new ProjectionSummaryBundle(p))
  projection := summary
  // The current-view frame is already checked above. Keep the explicit
  // assignment so the summary later retained by TRU has the render frame.
  projection.frameId := io.residencyIn.bits.propagation.update.frameId
  projection.center(0) := dot((0 until 3).map(summary.jPi(0)(_)), (0 until 3).map(state.mean(_)))
  projection.center(1) := dot((0 until 3).map(summary.jPi(1)(_)), (0 until 3).map(state.mean(_)))
  projection.depth := state.mean(2)
  val covariance = Wire(Vec(3, Vec(3, UInt(p.fpWidth.W))))
  covariance(0)(0) := state.covariance(0); covariance(0)(1) := state.covariance(1); covariance(0)(2) := state.covariance(2)
  covariance(1)(0) := state.covariance(1); covariance(1)(1) := state.covariance(3); covariance(1)(2) := state.covariance(4)
  covariance(2)(0) := state.covariance(2); covariance(2)(1) := state.covariance(4); covariance(2)(2) := state.covariance(5)
  val jSigma = Wire(Vec(2, Vec(3, UInt(p.fpWidth.W))))
  (0 until 2).foreach { row =>
    (0 until 3).foreach { column =>
      jSigma(row)(column) := dot((0 until 3).map(summary.jCov(row)(_)), (0 until 3).map(covariance(_)(column)))
    }
  }
  val screenCovariance = Wire(Vec(2, Vec(2, UInt(p.fpWidth.W))))
  (0 until 2).foreach { row =>
    (0 until 2).foreach { column =>
      screenCovariance(row)(column) := dot((0 until 3).map(jSigma(row)(_)), (0 until 3).map(summary.jCov(column)(_)))
    }
  }
  val covarianceDet = Fp32.sub(
    Fp32.mul(screenCovariance(0)(0), screenCovariance(1)(1)),
    Fp32.mul(screenCovariance(0)(1), screenCovariance(1)(0))
  )
  val support = Fp32.clampUnit(Fp32.mul(Fp32.abs(covarianceDet), "h41100000".U)) // Bounded support proxy from the paper equation.
  projection.opacity := state.opacity

  val lowOrderColor = Wire(Vec(3, UInt(p.fpWidth.W)))
  (0 until 3).foreach { channel =>
    lowOrderColor(channel) := (0 until 4).map { basis => Fp32.mul(state.sh(channel * p.shBases + basis), summary.shBasis(basis)) }.reduce(Fp32.add)
    // `b(v)` is the degree-0/1 SH estimate itself. The paper saturates P
    // after averaging the channelwise product, not each b(v) channel.
    projection.viewColor(channel) := lowOrderColor(channel)
  }
  val opacity = Fp32.clampUnit(Fp32.add(Fp32.Half, Fp32.mul(state.opacity, Fp32.Quarter)))
  val potentialRgb = (0 until 3).map(channel => Fp32.mul(Fp32.mul(opacity, support), projection.viewColor(channel)))
  val potential = Fp32.clampUnit(Fp32.mul(potentialRgb.reduce(Fp32.add), "h3eaaaaab".U))
  val kappa = Fp32.sub(Fp32.One, Fp32.clampUnit(io.residencyIn.bits.energy))
  val rho = Fp32.sub(Fp32.One, Fp32.mul(kappa, Fp32.sub(Fp32.One, potential)))
  val omega = Fp32.mul(rho, support)
  val structural = io.residencyIn.bits.propagation.update.structural =/= ScdGsConsts.StructuralNone.U
  // CIU receives only complete StateSPM responses. Candidate status permits
  // the cold-path history exception after that state boundary has completed.
  val insufficient = io.residencyIn.bits.sampleCount < p.siuWindow.U && io.residencyIn.bits.tier =/= ScdGsConsts.TierCandidate.U
  val fullPath = structural || insufficient || Fp32.compareLt(p.tauRho.U(p.fpWidth.W), rho)
  val filterCandidate = Fp32.compareLt(omega, p.tauD.U(p.fpWidth.W))
  val filterPath = !fullPath && filterCandidate && summary.requestKind === ScdGsConsts.RequestUser.U
  val path = Mux(fullPath, ScdGsConsts.PathFull.U, Mux(filterPath, ScdGsConsts.PathFilter.U, ScdGsConsts.PathLight.U))

  // A mismatched view or an incomplete StateSPM response is consumed and
  // counted instead of blocking a later valid pair. A valid complete pair
  // remains atomic with the issue output.
  val validPair = pairedInput && matchingView && completeState
  io.residencyIn.ready := io.currentViewIn.valid && (!validPair || io.issueOut.ready)
  io.currentViewIn.ready := io.residencyIn.valid && (!validPair || io.issueOut.ready)
  io.issueOut.valid := validPair
  io.issueOut.bits.residency := io.residencyIn.bits
  io.issueOut.bits.view := projection
  io.issueOut.bits.path := path
  io.issueOut.bits.supportArea := support
  io.issueOut.bits.colorEstimate := projection.viewColor
  io.issueOut.bits.renderingPotential := potential
  io.issueOut.bits.kappa := kappa
  io.issueOut.bits.rho := rho
  io.issueOut.bits.omega := omega
  when(io.issueOut.fire) {
    when(path === ScdGsConsts.PathFull.U) { full := full + 1.U }
    when(path === ScdGsConsts.PathLight.U) { light := light + 1.U }
    when(path === ScdGsConsts.PathFilter.U) { filter := filter + 1.U }
  }
  when(pairedInput && (!matchingView || !completeState)) { invalid := invalid + 1.U }
  io.fullCount := full
  io.lightCount := light
  io.filterCount := filter
  io.invalidInputCount := invalid
}
