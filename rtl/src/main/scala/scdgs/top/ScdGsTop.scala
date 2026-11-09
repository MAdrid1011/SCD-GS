package scdgs.top

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.ciu.ConditionedIssueUnit
import scdgs.config._
import scdgs.fru.FusedResidencyUnit
import scdgs.memory._
import scdgs.siu.SplatImpactUnit
import scdgs.tru.TieredRenderUnit
import scdgs.tuu.TrainingUpdateUnit

/** Public stream interface. Prior training summaries, current views, and state are explicit inputs. */
class ScdGsTopIO(p: ScdGsParams) extends Bundle {
  val canonicalIn = Flipped(Decoupled(new CanonicalUpdateBundle(p)))
  val priorTrainingSummaryIn = Flipped(Decoupled(new ProjectionSummaryBundle(p)))
  val currentViewIn = Flipped(Decoupled(new ProjectionSummaryBundle(p)))
  val stateSeedIn = Flipped(Decoupled(new StateSeedBundle(p)))
  val candidateFillIn = Flipped(Decoupled(new CandidateFillBundle(p)))
  val targetIn = Flipped(Decoupled(new TargetPixelBundle(p)))
  val frameEnd = Input(Bool())
  val candidateReq = Decoupled(new MemoryLineReqBundle(p))
  val metadataReadGid = Input(UInt(p.gidWidth.W))
  val metadataSnapshot = Output(new MetaEntryBundle(p))
  val metadataSnapshotValid = Output(Bool())
  val counters = Output(new MechanismCountersBundle)
}

/**
  * Mechanism-faithful public integration. Bootstrap prior-training metadata
  * enters through `priorTrainingSummaryIn`; current views pair with complete
  * residency records at CIU. Feedback updates reuse SplatSPM's latest
  * training-render summary.
  */
class ScdGsTop(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new ScdGsTopIO(p))
  val siu = Module(new SplatImpactUnit(p))
  val fru = Module(new FusedResidencyUnit(p))
  val ciu = Module(new ConditionedIssueUnit(p))
  val tru = Module(new TieredRenderUnit(p))
  val tuu = Module(new TrainingUpdateUnit(p))
  val state = Module(new StateSpm(p))
  val meta = Module(new MetaSpm(p))
  val splat = Module(new SplatSpm(p))
  val frame = Module(new FrameSpm(p))

  state.io.seedIn <> io.stateSeedIn
  state.io.candidateFillIn <> io.candidateFillIn

  val feedbackUpdate = tuu.io.updateOut.valid
  val selectedUpdate = Mux(feedbackUpdate, tuu.io.updateOut.bits, io.canonicalIn.bits)
  splat.io.summaryReadGid := selectedUpdate.gid
  val retainedSummaryAvailable = splat.io.summaryValid
  val feedbackHasSummary = feedbackUpdate && retainedSummaryAvailable
  // External summaries establish only bootstrap state. Once a training render
  // has retained a summary for this GID, every later external or feedback
  // update must use that record so an input cannot replace the prior view
  // consumed by SIU.
  val externalBootstrap = !feedbackUpdate && !retainedSummaryAvailable
  siu.io.updateIn.valid := feedbackHasSummary || (!feedbackUpdate && io.canonicalIn.valid)
  siu.io.updateIn.bits := selectedUpdate
  tuu.io.updateOut.ready := feedbackHasSummary && siu.io.updateIn.ready
  io.canonicalIn.ready := siu.io.updateIn.ready && !feedbackUpdate
  siu.io.priorSummaryIn.valid := Mux(
    feedbackUpdate,
    feedbackHasSummary,
    Mux(retainedSummaryAvailable, io.canonicalIn.valid, io.priorTrainingSummaryIn.valid)
  )
  siu.io.priorSummaryIn.bits := Mux(
    feedbackUpdate || retainedSummaryAvailable,
    splat.io.summaryRead,
    io.priorTrainingSummaryIn.bits
  )
  io.priorTrainingSummaryIn.ready := externalBootstrap && siu.io.priorSummaryIn.ready && io.canonicalIn.valid

  fru.io.commitIn <> siu.io.propagationOut
  fru.io.frameEnd := io.frameEnd

  // A render-read event is produced only by the transferred training
  // trajectory. There is no independent top-level render-read source.
  fru.io.renderReadIn <> tru.io.renderReadOut

  // Every residency record crosses StateSPM before CIU. This resolves the
  // physical partition selected by FRU and leaves a cold candidate record
  // pending until its matching complete state fill is present. A one-entry
  // handoff is intentional: it preserves the Decoupled transaction identity
  // across a potentially unbounded external refill delay.
  val lookupPending = RegInit(false.B)
  val pendingResidency = Reg(new ResidencyBundle(p))
  val resolvedValid = RegInit(false.B)
  val resolvedResidency = Reg(new ResidencyBundle(p))
  val canStartLookup = !lookupPending && !resolvedValid

  state.io.read.valid := fru.io.residencyOut.valid && canStartLookup
  state.io.read.bits.gid := fru.io.residencyOut.bits.propagation.update.gid
  state.io.read.bits.frameId := fru.io.residencyOut.bits.propagation.update.frameId
  state.io.read.bits.tier := fru.io.residencyOut.bits.tier
  state.io.read.bits.row := fru.io.residencyOut.bits.row
  fru.io.residencyOut.ready := canStartLookup && state.io.read.ready
  when(fru.io.residencyOut.fire) {
    pendingResidency := fru.io.residencyOut.bits
    lookupPending := true.B
  }

  state.io.readResp.ready := lookupPending && !resolvedValid
  val stateResolvedResidency = Wire(new ResidencyBundle(p))
  stateResolvedResidency := pendingResidency
  stateResolvedResidency.state := state.io.readResp.bits.state
  stateResolvedResidency.stateReady := true.B
  when(state.io.readResp.fire) {
    lookupPending := false.B
    when(state.io.readResp.bits.hit) {
      resolvedResidency := stateResolvedResidency
      resolvedValid := true.B
    }
  }
  io.candidateReq <> state.io.candidateReq
  fru.io.candidateMissIn := state.io.candidateReq.fire

  ciu.io.residencyIn.valid := resolvedValid
  ciu.io.residencyIn.bits := resolvedResidency
  when(ciu.io.residencyIn.fire) { resolvedValid := false.B }
  ciu.io.currentViewIn <> io.currentViewIn
  tru.io.issueIn <> ciu.io.issueOut
  tru.io.frameEnd := io.frameEnd

  // A completed training render has an independent direct render-read to FRU
  // and bounded trace deliveries to storage and TUU. The bridge preserves each
  // trace while allowing a consumer to backpressure without a same-cycle
  // fan-out rule.
  // A consumed malformed bootstrap pair must not become a prior-render summary.
  // Only SIU-accepted external pairs establish SplatSPM state.
  val bootstrapSummary = externalBootstrap && siu.io.acceptedInput
  val tracePending = RegInit(false.B)
  val pendingTrace = Reg(new RenderTraceBundle(p))
  val traceToTuuDelivered = RegInit(false.B)
  val traceToSplatDelivered = RegInit(false.B)
  val summaryToSplatDelivered = RegInit(false.B)
  val renderToFrameDelivered = RegInit(false.B)

  tru.io.traceOut.ready := !tracePending
  tuu.io.traceIn.valid := tracePending && !traceToTuuDelivered
  tuu.io.traceIn.bits := pendingTrace
  splat.io.traceWrite.valid := tracePending && !traceToSplatDelivered
  splat.io.traceWrite.bits := pendingTrace
  frame.io.renderWrite.valid := tracePending && !renderToFrameDelivered
  frame.io.renderWrite.bits.pixel := pendingTrace.pixel
  frame.io.renderWrite.bits.frameId := pendingTrace.frameId
  frame.io.renderWrite.bits.rgb := pendingTrace.color
  frame.io.renderWrite.bits.transmittance := pendingTrace.transmittance
  val traceSummaryWrite = tracePending && !summaryToSplatDelivered && !bootstrapSummary
  splat.io.summaryWrite.valid := bootstrapSummary || traceSummaryWrite
  splat.io.summaryWrite.bits := Mux(bootstrapSummary, io.priorTrainingSummaryIn.bits, pendingTrace.projection)

  val traceDeliveryComplete =
    (traceToTuuDelivered || tuu.io.traceIn.fire) &&
      (traceToSplatDelivered || splat.io.traceWrite.fire) &&
      (summaryToSplatDelivered || traceSummaryWrite && splat.io.summaryWrite.ready) &&
      (renderToFrameDelivered || frame.io.renderWrite.fire)
  when(tracePending) {
    when(traceDeliveryComplete) {
      tracePending := false.B
      traceToTuuDelivered := false.B
      traceToSplatDelivered := false.B
      summaryToSplatDelivered := false.B
      renderToFrameDelivered := false.B
    }.otherwise {
      when(tuu.io.traceIn.fire) { traceToTuuDelivered := true.B }
      when(splat.io.traceWrite.fire) { traceToSplatDelivered := true.B }
      when(traceSummaryWrite && splat.io.summaryWrite.ready) { summaryToSplatDelivered := true.B }
      when(frame.io.renderWrite.fire) { renderToFrameDelivered := true.B }
    }
  }.elsewhen(tru.io.traceOut.fire) {
    pendingTrace := tru.io.traceOut.bits
    tracePending := true.B
    traceToTuuDelivered := false.B
    traceToSplatDelivered := false.B
    summaryToSplatDelivered := false.B
    renderToFrameDelivered := false.B
  }

  // TUU owns a one-target buffer and may receive a target before TRU closes
  // the frame. A separate bounded fan-out retains that target until both the
  // TUU buffer and FrameSPM have accepted their own transfer.
  val targetPending = RegInit(false.B)
  val pendingTarget = Reg(new TargetPixelBundle(p))
  val targetToTuuDelivered = RegInit(false.B)
  val targetToFrameDelivered = RegInit(false.B)
  // Do not prefetch a second target while TUU owns its one target buffer.
  // The bridge is an input handoff, not an additional target queue.
  io.targetIn.ready := !targetPending && tuu.io.targetIn.ready
  tuu.io.targetIn.valid := targetPending && !targetToTuuDelivered
  tuu.io.targetIn.bits := pendingTarget
  frame.io.targetWrite.valid := targetPending && !targetToFrameDelivered
  frame.io.targetWrite.bits := pendingTarget
  val targetDeliveryComplete =
    (targetToTuuDelivered || tuu.io.targetIn.fire) &&
      (targetToFrameDelivered || frame.io.targetWrite.fire)
  when(targetPending) {
    when(targetDeliveryComplete) {
      targetPending := false.B
      targetToTuuDelivered := false.B
      targetToFrameDelivered := false.B
    }.otherwise {
      when(tuu.io.targetIn.fire) { targetToTuuDelivered := true.B }
      when(frame.io.targetWrite.fire) { targetToFrameDelivered := true.B }
    }
  }.elsewhen(io.targetIn.fire) {
    assert(tuu.io.targetIn.ready)
    pendingTarget := io.targetIn.bits
    targetPending := true.B
    targetToTuuDelivered := false.B
    targetToFrameDelivered := false.B
  }
  state.io.optimizerReadGid := tuu.io.optimizerReadGid
  tuu.io.optimizerState := state.io.optimizerState
  tuu.io.optimizerValid := state.io.optimizerValid
  tuu.io.stateWrite <> state.io.write

  // Publish the FRU's post-event directory state. Confirmation and
  // training-render-read events therefore remain visible even though they do
  // not all produce a residency record for StateSPM.
  meta.io.write.valid := fru.io.metadataSnapshot.valid
  meta.io.writeGid := fru.io.metadataSnapshotGid
  meta.io.write.bits := fru.io.metadataSnapshot.bits
  meta.io.readGid := io.metadataReadGid
  io.metadataSnapshot := meta.io.read
  io.metadataSnapshotValid := meta.io.readValid
  splat.io.traceReadGid := 0.U
  frame.io.targetReadPixel := 0.U
  frame.io.renderReadPixel := 0.U

  io.counters.siuCommit := siu.io.commitCount
  io.counters.siuConfirmation := siu.io.confirmationCount
  io.counters.fruActiveHit := fru.io.activeHitCount
  io.counters.fruResidentHit := fru.io.residentHitCount
  io.counters.fruCandidateRefill := fru.io.candidateRefillCount
  io.counters.ciuFull := ciu.io.fullCount
  io.counters.ciuLight := ciu.io.lightCount
  io.counters.ciuFilter := ciu.io.filterCount
  io.counters.truEarlyStop := tru.io.earlyStopCount
  io.counters.truShOps := tru.io.shOpCount
  io.counters.truCoverageOps := tru.io.coverageOpCount
  io.counters.truBlendOps := tru.io.blendOpCount
  io.counters.tuuUpdates := tuu.io.updateCount
  io.counters.queueStalls := tru.io.queueStallCount
  io.counters.invalidInput := siu.io.invalidInputCount + fru.io.invalidInputCount + state.io.invalidInputCount +
    ciu.io.invalidInputCount + tru.io.invalidInputCount + tuu.io.invalidPairCount
}

/** Named public wrapper used by the SystemVerilog emitter. */
class ScdGsPublicTop(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new ScdGsTopIO(p))
  val core = Module(new ScdGsTop(p))
  core.io <> io
}
