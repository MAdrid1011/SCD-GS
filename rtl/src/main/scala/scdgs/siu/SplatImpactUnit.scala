package scdgs.siu

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config._
import scdgs.fp.Fp32
import scdgs.math.{DivF32, SqrtF32}

/** Shared 3x3 binary32 FMA array used by the SIU projection microsequence. */
class SharedFma3x3 extends Module {
  val io = IO(new Bundle {
    val matrix = Input(Vec(3, Vec(3, UInt(32.W))))
    val vector = Input(Vec(3, UInt(32.W)))
    val out = Output(Vec(3, UInt(32.W)))
  })
  (0 until 3).foreach { row =>
    val first = Fp32.fma(io.matrix(row)(0), io.vector(0), Fp32.Zero)
    val second = Fp32.fma(io.matrix(row)(1), io.vector(1), first)
    io.out(row) := Fp32.fma(io.matrix(row)(2), io.vector(2), second)
  }
}

/** Public binary32 activation used by SIU and coverage alpha formation. */
class ActivationUnit extends Module {
  val io = IO(new Bundle { val in = Input(UInt(32.W)); val out = Output(UInt(32.W)) })
  io.out := Fp32.clampUnit(Fp32.add(Fp32.Half, Fp32.mul(io.in, Fp32.Quarter)))
}

/** Evaluates full-order delta SH against the basis retained by the prior render. */
class ShEvalUnit(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val deltaSh = Input(Vec(p.shCoefficients, UInt(p.fpWidth.W)))
    val basis = Input(Vec(p.shBases, UInt(p.fpWidth.W)))
    val luminance = Output(UInt(p.fpWidth.W))
  })
  val rgb = Wire(Vec(3, UInt(p.fpWidth.W)))
  (0 until 3).foreach { channel =>
    rgb(channel) := (0 until p.shBases).map { basis => Fp32.mul(io.deltaSh(channel * p.shBases + basis), io.basis(basis)) }.reduce(Fp32.add)
  }
  val weighted = Fp32.add(Fp32.add(Fp32.mul(rgb(0), "h3e59b3d0".U), Fp32.mul(rgb(1), "h3f371759".U)), Fp32.mul(rgb(2), "h3d93dd98".U))
  io.luminance := Fp32.abs(weighted)
}

/** Stage I with explicit camera/Jacobian and SH contractions. */
class SplatImpactUnit(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val updateIn = Flipped(Decoupled(new CanonicalUpdateBundle(p)))
    val priorSummaryIn = Flipped(Decoupled(new ProjectionSummaryBundle(p)))
    val propagationOut = Decoupled(new PropagationBundle(p))
    // Distinguishes a consumed pair from a pair that passed SIU validation.
    // ScdGsTop uses this pulse to make bootstrap SplatSPM writes transactional.
    val acceptedInput = Output(Bool())
    val commitCount = Output(UInt(32.W))
    val confirmationCount = Output(UInt(32.W))
    val invalidInputCount = Output(UInt(32.W))
  })

  private val gidIndexWidth = math.max(1, log2Ceil(p.maxGaussians))
  private def index(gid: UInt): UInt = if (p.maxGaussians == 1) 0.U else gid(gidIndexWidth - 1, 0)
  private def valid(gid: UInt): Bool = gid < p.maxGaussians.U
  private def validStructural(marker: UInt): Bool = marker <= ScdGsConsts.StructuralReset.U
  private def dot(left: Seq[UInt], right: Seq[UInt]): UInt = left.zip(right).map { case (a, b) => Fp32.mul(a, b) }.reduce(Fp32.add)

  val fma3x3 = Module(new SharedFma3x3)
  val norm = Module(new SqrtF32)
  val divide = Module(new DivF32)
  val oldActivation = Module(new ActivationUnit)
  val newActivation = Module(new ActivationUnit)
  val shEval = Module(new ShEvalUnit(p))
  val pending = RegInit(false.B)
  val update = Reg(new CanonicalUpdateBundle(p))
  val summary = Reg(new ProjectionSummaryBundle(p))
  val history = RegInit(VecInit(Seq.fill(p.maxGaussians)(VecInit(Seq.fill(p.siuWindow)(Fp32.Zero)))))
  val historyHead = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(log2Ceil(p.siuWindow).W))))
  val historyCount = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(log2Ceil(p.siuWindow + 1).W))))
  val historySum = RegInit(VecInit(Seq.fill(p.maxGaussians)(Fp32.Zero)))
  val historyFrame = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(p.frameWidth.W))))
  val historyFrameValid = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val capturedDeltaU = Reg(UInt(p.fpWidth.W))
  val capturedDeltaV = Reg(UInt(p.fpWidth.W))
  val commits = RegInit(0.U(32.W))
  val confirmations = RegInit(0.U(32.W))
  val invalid = RegInit(0.U(32.W))

  val canAccept = !pending && norm.io.in.ready
  io.updateIn.ready := canAccept && io.priorSummaryIn.valid
  io.priorSummaryIn.ready := canAccept && io.updateIn.valid
  val inputAccepted = io.updateIn.fire && io.priorSummaryIn.fire
  val inputGidValid = valid(io.updateIn.bits.gid)
  val priorFrameValid = inputGidValid && historyFrameValid(index(io.updateIn.bits.gid))
  val monotonicSourceFrame = !priorFrameValid || io.updateIn.bits.frameId > historyFrame(index(io.updateIn.bits.gid))
  val inputWellFormed = inputGidValid &&
    io.updateIn.bits.gid === io.priorSummaryIn.bits.gid &&
    io.priorSummaryIn.bits.frameId < io.updateIn.bits.frameId &&
    io.priorSummaryIn.bits.requestKind === ScdGsConsts.RequestTraining.U &&
    Fp32.isPositiveNormal(io.priorSummaryIn.bits.radius) &&
    validStructural(io.updateIn.bits.structural) &&
    monotonicSourceFrame
  val inputFire = inputAccepted && inputWellFormed
  io.acceptedInput := inputFire

  fma3x3.io.matrix := io.priorSummaryIn.bits.rCam
  fma3x3.io.vector := io.updateIn.bits.deltaMu
  val deltaU = dot((0 until 3).map(io.priorSummaryIn.bits.jPi(0)(_)), (0 until 3).map(fma3x3.io.out(_)))
  val deltaV = dot((0 until 3).map(io.priorSummaryIn.bits.jPi(1)(_)), (0 until 3).map(fma3x3.io.out(_)))

  val sigma = Wire(Vec(3, Vec(3, UInt(p.fpWidth.W))))
  sigma(0)(0) := io.updateIn.bits.deltaSigma(0); sigma(0)(1) := io.updateIn.bits.deltaSigma(1); sigma(0)(2) := io.updateIn.bits.deltaSigma(2)
  sigma(1)(0) := io.updateIn.bits.deltaSigma(1); sigma(1)(1) := io.updateIn.bits.deltaSigma(3); sigma(1)(2) := io.updateIn.bits.deltaSigma(4)
  sigma(2)(0) := io.updateIn.bits.deltaSigma(2); sigma(2)(1) := io.updateIn.bits.deltaSigma(4); sigma(2)(2) := io.updateIn.bits.deltaSigma(5)
  val jSigma = Wire(Vec(2, Vec(3, UInt(p.fpWidth.W))))
  (0 until 2).foreach { row => (0 until 3).foreach { column => jSigma(row)(column) := dot((0 until 3).map(io.priorSummaryIn.bits.jCov(row)(_)), (0 until 3).map(sigma(_)(column))) } }
  val projectedCov = Wire(Vec(2, Vec(2, UInt(p.fpWidth.W))))
  (0 until 2).foreach { row => (0 until 2).foreach { column => projectedCov(row)(column) := dot((0 until 3).map(jSigma(row)(_)), (0 until 3).map(io.priorSummaryIn.bits.jCov(column)(_))) } }
  val covarianceNorm = (for (row <- 0 until 2; column <- 0 until 2) yield Fp32.mul(projectedCov(row)(column), projectedCov(row)(column))).reduce(Fp32.add)
  norm.io.in.valid := inputFire
  norm.io.in.bits := covarianceNorm

  when(inputAccepted) {
    when(inputWellFormed) {
      update := io.updateIn.bits
      summary := io.priorSummaryIn.bits
      capturedDeltaU := deltaU
      capturedDeltaV := deltaV
      pending := true.B
    }.otherwise {
      invalid := invalid + 1.U
    }
  }

  oldActivation.io.in := summary.opacity
  newActivation.io.in := Fp32.add(summary.opacity, update.deltaOpacity)
  shEval.io.deltaSh := update.deltaSh
  shEval.io.basis := summary.shBasis
  val gidValid = valid(update.gid)
  val old = Mux(gidValid, history(index(update.gid))(historyHead(index(update.gid))), Fp32.Zero)
  val currentCount = Mux(gidValid, historyCount(index(update.gid)), 0.U)
  val currentSum = Mux(gidValid, historySum(index(update.gid)), Fp32.Zero)
  // The tile area and observation count are elaboration-time constants. Keep
  // their correctly rounded binary32 reciprocals rather than adding an
  // unsupported runtime approximation to the paper equations.
  val tileAreaScale = BigInt(java.lang.Float.floatToRawIntBits(1.0f / p.tilePixels.toFloat).toLong & 0xffffffffL).U(32.W)
  val coverage = Fp32.clampUnit(Fp32.mul(summary.area, tileAreaScale))
  divide.io.in.valid := pending && norm.io.out.valid
  divide.io.in.bits.numerator := Fp32.add(Fp32.add(Fp32.abs(capturedDeltaU), Fp32.abs(capturedDeltaV)), norm.io.out.bits)
  divide.io.in.bits.denominator := summary.radius
  val geometry = divide.io.out.bits
  val deltaAlpha = Fp32.abs(Fp32.sub(newActivation.io.out, oldActivation.io.out))
  val psi = Fp32.mul(coverage, Fp32.add(Fp32.add(geometry, deltaAlpha), shEval.io.luminance))
  val sameFrame = gidValid && historyFrameValid(index(update.gid)) && historyFrame(index(update.gid)) === update.frameId
  val fullWindow = currentCount === p.siuWindow.U
  val nextSum = Mux(sameFrame, currentSum, Fp32.add(Fp32.sub(currentSum, Mux(fullWindow, old, Fp32.Zero)), psi))
  val nextCount = Mux(sameFrame, currentCount, Mux(fullWindow, currentCount, currentCount + 1.U))
  val inverseCount = MuxLookup(nextCount, Fp32.Zero)(Seq(
    1.U -> "h3f800000".U(32.W),
    2.U -> "h3f000000".U(32.W),
    3.U -> "h3eaaaaab".U(32.W),
    4.U -> "h3e800000".U(32.W),
    5.U -> "h3e4ccccd".U(32.W)
  ))
  val phi = Fp32.mul(nextSum, inverseCount)
  val structural = update.structural =/= ScdGsConsts.StructuralNone.U
  val protectedWarmup = currentCount < (p.siuWindow - 1).U
  val commit = structural || protectedWarmup || Fp32.compareGe(phi, p.tauPhi.U(p.fpWidth.W))

  io.propagationOut.valid := pending && divide.io.out.valid
  io.propagationOut.bits.update := update
  io.propagationOut.bits.projection := summary
  io.propagationOut.bits.psi := psi
  io.propagationOut.bits.phi := phi
  io.propagationOut.bits.effectiveStrength := Mux(commit, update.strength, Fp32.Zero)
  io.propagationOut.bits.propagation := Mux(commit, ScdGsConsts.PropagationCommit.U, ScdGsConsts.PropagationConfirmation.U)
  norm.io.out.ready := pending && divide.io.in.ready
  divide.io.out.ready := pending && io.propagationOut.ready
  when(io.propagationOut.fire) {
    pending := false.B
    when(gidValid) {
      when(!sameFrame) {
        history(index(update.gid))(historyHead(index(update.gid))) := psi
        historyHead(index(update.gid)) := Mux(historyHead(index(update.gid)) === (p.siuWindow - 1).U, 0.U, historyHead(index(update.gid) + 1.U))
      }.otherwise {
        // The public transaction contract permits one update per gid/frame.
        // Keep the first accepted observation and flag duplicates below.
        invalid := invalid + 1.U
      }
      historySum(index(update.gid)) := nextSum
      historyCount(index(update.gid)) := nextCount
      historyFrame(index(update.gid)) := update.frameId
      historyFrameValid(index(update.gid)) := true.B
    }
    when(commit) { commits := commits + 1.U }.otherwise { confirmations := confirmations + 1.U }
  }
  io.commitCount := commits
  io.confirmationCount := confirmations
  io.invalidInputCount := invalid
}
