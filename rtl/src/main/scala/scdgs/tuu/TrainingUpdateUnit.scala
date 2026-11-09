package scdgs.tuu

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config._
import scdgs.fp.Fp32
import scdgs.math.{DivF32, SqrtF32}

/**
  * Stage V: pair a training trajectory with its target, form a scalar
 * surrogate update, update the full StateSPM record, and then return a
 * canonical attribute update to SIU. The public reference keeps one target
 * buffer so a target may arrive before the corresponding render trace.
  */
class TrainingUpdateUnit(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val traceIn = Flipped(Decoupled(new RenderTraceBundle(p)))
    val targetIn = Flipped(Decoupled(new TargetPixelBundle(p)))
    val optimizerReadGid = Output(UInt(p.gidWidth.W))
    val optimizerState = Input(new OptimizerStateBundle(p))
    val optimizerValid = Input(Bool())
    val updateOut = Decoupled(new CanonicalUpdateBundle(p))
    val stateWrite = Decoupled(new StateWriteBundle(p))
    val updateCount = Output(UInt(32.W))
    val invalidPairCount = Output(UInt(32.W))
  })

  private val laneSelectWidth = math.max(1, log2Ceil(p.trainingLanes))
  private def absFp(value: UInt): UInt = Mux(value(31), Fp32.negate(value), value)

  val updates = RegInit(0.U(32.W))
  val invalidPairs = RegInit(0.U(32.W))
  val targetValid = RegInit(false.B)
  val target = Reg(new TargetPixelBundle(p))
  val pending = RegInit(false.B)
  val updateValid = RegInit(false.B)
  val writeValid = RegInit(false.B)
  val updateRecord = Reg(new CanonicalUpdateBundle(p))
  val writeRecord = Reg(new StateWriteBundle(p))
  val stagedGid = Reg(UInt(p.gidWidth.W))
  val stagedFrame = Reg(UInt(p.frameWidth.W))
  val stagedState = Reg(new GaussianStateBundle(p))
  val stagedMoment = Reg(Vec(p.attributeCount, UInt(p.fpWidth.W)))
  val stagedVariance = Reg(Vec(p.attributeCount, UInt(p.fpWidth.W)))
  val sqrt = Module(new SqrtF32)
  val divide = Module(new DivF32)

  io.targetIn.ready := !targetValid && !pending && !updateValid && !writeValid
  when(io.targetIn.fire) {
    target := io.targetIn.bits
    targetValid := true.B
  }

  val traceGidValid = io.traceIn.bits.gid < p.maxGaussians.U
  // TRU emits training trajectories only for full or light work. A filter tag
  // or user-view record is a malformed direct TUU input and must not create
  // feedback state.
  val tracePathValid = io.traceIn.bits.path === ScdGsConsts.PathFull.U ||
    io.traceIn.bits.path === ScdGsConsts.PathLight.U
  val traceIsTraining = io.traceIn.bits.projection.requestKind === ScdGsConsts.RequestTraining.U
  // A matched trajectory feeds SIU in the next frame. The terminal unsigned
  // frame value has no representable successor and must not wrap to frame 0.
  val traceFrameHasSuccessor = io.traceIn.bits.frameId =/= p.maxFrameId.U(p.frameWidth.W)
  val matching = targetValid &&
    io.traceIn.bits.pixel === target.pixel &&
    io.traceIn.bits.frameId === target.frameId &&
    traceGidValid && tracePathValid && traceIsTraining && traceFrameHasSuccessor && io.optimizerValid &&
    io.optimizerState.gid === io.traceIn.bits.gid
  val canAcceptTrace = targetValid && !pending && !updateValid && !writeValid && sqrt.io.in.ready
  io.traceIn.ready := canAcceptTrace
  val pairFire = io.traceIn.fire
  io.optimizerReadGid := io.traceIn.bits.gid

  val residual = Fp32.mul(
    Fp32.add(
      Fp32.add(
        absFp(Fp32.sub(io.traceIn.bits.color(0), target.rgb(0))),
        absFp(Fp32.sub(io.traceIn.bits.color(1), target.rgb(1)))
      ),
      absFp(Fp32.sub(io.traceIn.bits.color(2), target.rgb(2)))
    ),
    "h3eaaaaab".U(32.W)
  )
  val selectedLane = io.traceIn.bits.pixel(laneSelectWidth - 1, 0)
  val lossLanes = Wire(Vec(p.trainingLanes, UInt(p.fpWidth.W)))
  val backwardLanes = Wire(Vec(p.trainingLanes, UInt(p.fpWidth.W)))
  (0 until p.trainingLanes).foreach { lane =>
    lossLanes(lane) := Mux(selectedLane === lane.U(laneSelectWidth.W), residual, Fp32.Zero)
    backwardLanes(lane) := Fp32.mul(
      Fp32.mul(lossLanes(lane), io.traceIn.bits.coverage),
      io.traceIn.bits.transmittance
    )
  }
  val localGradient = backwardLanes.reduce(Fp32.add)
  val surrogateGradient = localGradient

  val beta1 = p.adamBeta1.U(p.fpWidth.W)
  val beta2 = p.adamBeta2.U(p.fpWidth.W)
  val nextMoments = Wire(Vec(p.attributeCount, UInt(p.fpWidth.W)))
  val nextVariances = Wire(Vec(p.attributeCount, UInt(p.fpWidth.W)))
  (0 until p.attributeCount).foreach { attribute =>
    nextMoments(attribute) := Fp32.add(
      Fp32.mul(beta1, io.optimizerState.adamMoment(attribute)),
      Fp32.mul(Fp32.sub(Fp32.One, beta1), surrogateGradient)
    )
    nextVariances(attribute) := Fp32.add(
      Fp32.mul(beta2, io.optimizerState.adamVariance(attribute)),
      Fp32.mul(Fp32.sub(Fp32.One, beta2), Fp32.mul(surrogateGradient, surrogateGradient))
    )
  }

  sqrt.io.in.valid := pairFire && matching
  sqrt.io.in.bits := nextVariances(0)
  when(pairFire) {
    when(matching) {
      targetValid := false.B
      stagedGid := io.traceIn.bits.gid
      stagedFrame := io.traceIn.bits.frameId
      stagedState := io.optimizerState.state
      stagedMoment := nextMoments
      stagedVariance := nextVariances
      pending := true.B
    }.otherwise {
      invalidPairs := invalidPairs + 1.U
    }
  }

  divide.io.in.valid := pending && sqrt.io.out.valid && !updateValid && !writeValid
  divide.io.in.bits.numerator := Fp32.mul(p.adamLearningRate.U(p.fpWidth.W), stagedMoment(0))
  divide.io.in.bits.denominator := Fp32.add(sqrt.io.out.bits, p.adamEpsilon.U(p.fpWidth.W))
  val adamStep = divide.io.out.bits
  val negativeStep = Fp32.negate(adamStep)
  val nextState = Wire(new GaussianStateBundle(p))
  nextState := stagedState
  nextState.gid := stagedGid
  nextState.mean(0) := Fp32.add(stagedState.mean(0), negativeStep)
  nextState.mean(1) := Fp32.add(stagedState.mean(1), Fp32.mul(negativeStep, Fp32.Half))
  nextState.mean(2) := stagedState.mean(2)
  (0 until 6).foreach { coordinate =>
    nextState.covariance(coordinate) := Mux(
      coordinate.U === 0.U || coordinate.U === 3.U || coordinate.U === 5.U,
      Fp32.add(stagedState.covariance(coordinate), negativeStep),
      stagedState.covariance(coordinate)
    )
  }
  nextState.opacity := Fp32.add(stagedState.opacity, Fp32.mul(negativeStep, Fp32.Quarter))
  (0 until p.shCoefficients).foreach { coefficient =>
    nextState.sh(coefficient) := Fp32.add(stagedState.sh(coefficient), Fp32.mul(negativeStep, Fp32.Half))
  }

  sqrt.io.out.ready := pending && !updateValid && !writeValid && divide.io.in.ready
  divide.io.out.ready := pending && !updateValid && !writeValid
  when(divide.io.out.fire) {
    updateRecord := 0.U.asTypeOf(new CanonicalUpdateBundle(p))
    updateRecord.gid := stagedGid
    updateRecord.frameId := stagedFrame + 1.U
    updateRecord.deltaMu(0) := negativeStep
    updateRecord.deltaMu(1) := Fp32.mul(negativeStep, Fp32.Half)
    updateRecord.deltaMu(2) := Fp32.Zero
    (0 until 6).foreach { coordinate =>
      updateRecord.deltaSigma(coordinate) := Mux(
        coordinate.U === 0.U || coordinate.U === 3.U || coordinate.U === 5.U,
        negativeStep,
        Fp32.Zero
      )
    }
    updateRecord.deltaOpacity := Fp32.mul(negativeStep, Fp32.Quarter)
    (0 until p.shCoefficients).foreach { coefficient =>
      updateRecord.deltaSh(coefficient) := Fp32.mul(negativeStep, Fp32.Half)
    }
    updateRecord.strength := Fp32.clampUnit(absFp(adamStep))
    updateRecord.structural := ScdGsConsts.StructuralNone.U

    writeRecord := 0.U.asTypeOf(new StateWriteBundle(p))
    writeRecord.gid := stagedGid
    writeRecord.state := nextState
    writeRecord.adamMoment := stagedMoment
    writeRecord.adamVariance := stagedVariance
    writeValid := true.B
    pending := false.B
  }

  io.updateOut.valid := updateValid
  io.updateOut.bits := updateRecord
  io.stateWrite.valid := writeValid
  io.stateWrite.bits := writeRecord
  // Persist the state transition before exposing its feedback record. Without
  // this ordering, a stalled StateSPM write could let the next SIU event read
  // the pre-update Gaussian through the feedback path.
  when(io.stateWrite.fire) {
    writeValid := false.B
    updateValid := true.B
  }
  when(io.updateOut.fire) {
    updateValid := false.B
    updates := updates + 1.U
  }
  io.updateCount := updates
  io.invalidPairCount := invalidPairs
}
