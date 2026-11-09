package scdgs.tru

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config._
import scdgs.fp.Fp32

/**
  * Stage IV records issues for one frame in a bounded sorted buffer. At the
  * frame boundary it drains the globally earliest `(tile, depth, gid)` record,
  * not merely the head of a residency queue. Residency only controls issue and
  * refill order; it never changes alpha-compositing order.
  */
class TieredRenderUnit(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val issueIn = Flipped(Decoupled(new IssueBundle(p)))
    val frameEnd = Input(Bool())
    val traceOut = Decoupled(new RenderTraceBundle(p))
    val renderReadOut = Decoupled(new RenderReadBundle(p))
    val earlyStopCount = Output(UInt(32.W))
    val queueStallCount = Output(UInt(32.W))
    val invalidInputCount = Output(UInt(32.W))
    val shOpCount = Output(UInt(32.W))
    val coverageOpCount = Output(UInt(32.W))
    val blendOpCount = Output(UInt(32.W))
  })

  private val slotWidth = math.max(1, log2Ceil(p.renderQueueDepth))
  private val tileEntries = 1 << p.tileWidth

  val entries = Reg(Vec(p.renderQueueDepth, new IssueBundle(p)))
  val valid = RegInit(VecInit(Seq.fill(p.renderQueueDepth)(false.B)))
  val draining = RegInit(false.B)
  // A frame boundary has no identifier, so keep one explicit frame identity
  // for the queued batch. A later frame must wait for this batch to drain.
  val bufferedFrameValid = RegInit(false.B)
  val bufferedFrameId = Reg(UInt(p.frameWidth.W))
  val blendOrder = RegInit(0.U(16.W))
  val tileTransmittance = RegInit(VecInit(Seq.fill(tileEntries)(Fp32.One)))
  val tileFrame = RegInit(VecInit(Seq.fill(tileEntries)(0.U(p.frameWidth.W))))
  val tileSeen = RegInit(VecInit(Seq.fill(tileEntries)(false.B)))
  val earlyStops = RegInit(0.U(32.W))
  val stalls = RegInit(0.U(32.W))
  val invalidInputs = RegInit(0.U(32.W))
  val shOps = RegInit(0.U(32.W))
  val coverageOps = RegInit(0.U(32.W))
  val blendOps = RegInit(0.U(32.W))
  // The trajectory and the FRU render-read have distinct ready/valid links.
  // Retain delivery state for the selected training record so either consumer
  // may be backpressured without duplicating the other record.
  val traceDelivered = RegInit(false.B)
  val renderReadDelivered = RegInit(false.B)
  val frameEndWasHigh = RegNext(io.frameEnd, false.B)
  val frameEndPulse = io.frameEnd && !frameEndWasHigh

  def earlier(left: IssueBundle, right: IssueBundle): Bool = {
    val leftProjection = left.view
    val rightProjection = right.view
      leftProjection.tile < rightProjection.tile ||
      (leftProjection.tile === rightProjection.tile &&
        (Fp32.compareLt(leftProjection.depth, rightProjection.depth) ||
          (Fp32.compareEq(leftProjection.depth, rightProjection.depth) &&
            left.residency.propagation.update.gid < right.residency.propagation.update.gid)))
  }

  val freeMask = VecInit(valid.map(entryValid => !entryValid)).asUInt
  val hasFree = freeMask.orR
  val freeIndex = PriorityEncoder(freeMask)
  val incomingFrame = io.issueIn.bits.residency.propagation.update.frameId
  val incomingTraining = io.issueIn.bits.view.requestKind === ScdGsConsts.RequestTraining.U
  val incomingPathKnown = io.issueIn.bits.path === ScdGsConsts.PathFull.U ||
    io.issueIn.bits.path === ScdGsConsts.PathLight.U ||
    io.issueIn.bits.path === ScdGsConsts.PathFilter.U
  val incomingInvalidPath = !incomingPathKnown
  // A training trajectory must feed a canonical update in the next frame.
  // Reject the terminal frame at the producer boundary so it cannot create a
  // trace, render-read, or retained render state that TUU must later discard.
  // Terminal-frame user work remains legal because it has no feedback edge.
  val incomingTerminalTraining = incomingTraining && incomingFrame === p.maxFrameId.U(p.frameWidth.W)
  // The global `(tile, depth, gid)` order is defined only for numerical depth
  // values. Reject NaN and infinity before they can make the comparator's
  // outcome depend on buffer insertion order.
  val incomingNonFiniteDepth = !Fp32.isFinite(io.issueIn.bits.view.depth)
  val incomingFilteredUser = io.issueIn.bits.path === ScdGsConsts.PathFilter.U && !incomingTraining
  val frameMatchesBufferedBatch = !bufferedFrameValid || incomingFrame === bufferedFrameId
  io.issueIn.ready := !draining && (
    incomingInvalidPath || incomingTerminalTraining || incomingNonFiniteDepth || incomingFilteredUser ||
      (hasFree && frameMatchesBufferedBatch)
  )
  when(io.issueIn.fire) {
    when(incomingInvalidPath || incomingTerminalTraining || incomingNonFiniteDepth) {
      invalidInputs := invalidInputs + 1.U
    }.elsewhen(!incomingFilteredUser) {
      entries(freeIndex) := io.issueIn.bits
      valid(freeIndex) := true.B
      when(!bufferedFrameValid) {
        bufferedFrameId := incomingFrame
        bufferedFrameValid := true.B
      }
    }
  }
  when(io.issueIn.valid && !io.issueIn.ready) { stalls := stalls + 1.U }

  val initialIndex = 0.U(slotWidth.W)
  var bestIndex: UInt = initialIndex
  var bestValid: Bool = valid(0)
  (1 until p.renderQueueDepth).foreach { slot =>
    val take = valid(slot) && (!bestValid || earlier(entries(slot), entries(bestIndex)))
    bestIndex = Mux(take, slot.U(slotWidth.W), bestIndex)
    bestValid = bestValid || valid(slot)
  }
  val selected = entries(bestIndex)
  val selectedProjection = selected.view
  val selectedFrame = selected.residency.propagation.update.frameId
  val selectedTile = selectedProjection.tile
  val currentTransmittance = Mux(
    tileSeen(selectedTile) && tileFrame(selectedTile) === selectedFrame,
    tileTransmittance(selectedTile),
    Fp32.One
  )
  val stopped = Fp32.compareLt(currentTransmittance, p.transmittanceStop.U(p.fpWidth.W))
  val selectedTraining = selectedProjection.requestKind === ScdGsConsts.RequestTraining.U
  val selectedFiltered = selected.path === ScdGsConsts.PathFilter.U && !selectedTraining
  val effectivePath = Mux(
    selected.path === ScdGsConsts.PathFilter.U && selectedTraining,
    ScdGsConsts.PathLight.U,
    selected.path
  )

  val opacity = Fp32.clampUnit(Fp32.add(Fp32.Half, Fp32.mul(selected.residency.state.opacity, Fp32.Quarter)))
  val coverage = Fp32.clampUnit(selected.supportArea)
  val alpha = Fp32.clampUnit(Fp32.mul(opacity, coverage))
  val nextTransmittance = Fp32.mul(currentTransmittance, Fp32.sub(Fp32.One, alpha))
  val fullColor = Wire(Vec(3, UInt(p.fpWidth.W)))
  val lightColor = Wire(Vec(3, UInt(p.fpWidth.W)))
  (0 until 3).foreach { channel =>
    fullColor(channel) := (0 until p.shBases).map { basis =>
      Fp32.mul(selected.residency.state.sh(channel * p.shBases + basis), selectedProjection.shBasis(basis))
    }.reduce(Fp32.add)
    lightColor(channel) := Fp32.fp16ToFp32(Fp32.fp32ToFp16(selected.colorEstimate(channel)))
  }
  val color = Wire(Vec(3, UInt(p.fpWidth.W)))
  (0 until 3).foreach { channel =>
    color(channel) := Mux(effectivePath === ScdGsConsts.PathFull.U, fullColor(channel), lightColor(channel))
  }

  val emitTrainingOutputs = draining && bestValid && !stopped && !selectedFiltered && selectedTraining
  io.traceOut.valid := emitTrainingOutputs && !traceDelivered
  io.traceOut.bits := 0.U.asTypeOf(new RenderTraceBundle(p))
  io.traceOut.bits.gid := selected.residency.propagation.update.gid
  io.traceOut.bits.frameId := selectedFrame
  // This public proxy renderer has no raster kernel. A zero-extended tile ID
  // is therefore the representative-pixel key for the trace, FrameSPM, and
  // the matching target transaction.
  io.traceOut.bits.pixel := Cat(0.U((32 - p.tileWidth).W), selectedTile)
  io.traceOut.bits.blendOrder := blendOrder
  io.traceOut.bits.coverage := coverage
  io.traceOut.bits.transmittance := nextTransmittance
  io.traceOut.bits.color := color
  io.traceOut.bits.projection := selectedProjection
  io.traceOut.bits.path := effectivePath
  io.renderReadOut.valid := emitTrainingOutputs && !renderReadDelivered
  io.renderReadOut.bits.gid := selected.residency.propagation.update.gid
  io.renderReadOut.bits.frameId := selectedFrame

  val traceComplete = traceDelivered || io.traceOut.fire
  val renderReadComplete = renderReadDelivered || io.renderReadOut.fire
  val outputReady = !emitTrainingOutputs || (traceComplete && renderReadComplete)
  val consume = draining && bestValid && (stopped || selectedFiltered || outputReady)
  when(consume) {
    valid(bestIndex) := false.B
    when(stopped) {
      earlyStops := earlyStops + 1.U
    }.elsewhen(!selectedFiltered) {
      tileTransmittance(selectedTile) := nextTransmittance
      tileFrame(selectedTile) := selectedFrame
      tileSeen(selectedTile) := true.B
      blendOrder := blendOrder + 1.U
      coverageOps := coverageOps + 1.U
      blendOps := blendOps + 1.U
      when(effectivePath === ScdGsConsts.PathFull.U || effectivePath === ScdGsConsts.PathLight.U) {
        shOps := shOps + 1.U
      }
    }
    traceDelivered := false.B
    renderReadDelivered := false.B
  }.otherwise {
    when(io.traceOut.fire) { traceDelivered := true.B }
    when(io.renderReadOut.fire) { renderReadDelivered := true.B }
  }

  // `frameEnd` has no ready signal. While a closed frame is still draining,
  // no new issue can enter the one-frame buffer, so a later empty boundary
  // must not reset the current frame's global blend order.
  when(frameEndPulse && !draining) {
    draining := true.B
    blendOrder := 0.U
  }
  when(draining && !bestValid) {
    draining := false.B
    bufferedFrameValid := false.B
  }
  io.earlyStopCount := earlyStops
  io.queueStallCount := stalls
  io.invalidInputCount := invalidInputs
  io.shOpCount := shOps
  io.coverageOpCount := coverageOps
  io.blendOpCount := blendOps
}
