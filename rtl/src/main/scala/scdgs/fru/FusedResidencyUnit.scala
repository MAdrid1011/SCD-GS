package scdgs.fru

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config._
import scdgs.fp.Fp32

/** Stage II lookup arbiter, direct-indexed tier directory, and rank allocator. */
class FusedResidencyUnit(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val commitIn = Flipped(Decoupled(new PropagationBundle(p)))
    val renderReadIn = Flipped(Decoupled(new RenderReadBundle(p)))
    // StateSPM reports an actual cold candidate miss only when it accepts the
    // aligned line request. This keeps the public refill counter distinct
    // from a candidate-tier lookup that already has a filled payload.
    val candidateMissIn = Input(Bool())
    val frameEnd = Input(Bool())
    val residencyOut = Decoupled(new ResidencyBundle(p))
    val activeHitCount = Output(UInt(32.W))
    val residentHitCount = Output(UInt(32.W))
    val candidateRefillCount = Output(UInt(32.W))
    val invalidInputCount = Output(UInt(32.W))
    // MetaSPM receives the post-event directory state. This covers commits,
    // confirmations, and training-render reads rather than only records that
    // continue to StateSPM.
    val metadataSnapshot = Output(Valid(new MetaEntryBundle(p)))
    val metadataSnapshotGid = Output(UInt(p.gidWidth.W))
  })

  private val indexWidth = math.max(1, log2Ceil(p.maxGaussians))
  private val frameCountWidth = math.max(1, log2Ceil(p.siuWindow))
  private val activeSlots = math.min(p.activeCapacity, p.maxGaussians)
  private val residentSlots = math.min(p.residentCapacity, p.maxGaussians - activeSlots)
  private val selectionSlots = activeSlots + residentSlots
  // ScdGsParams requires both StateSPM partitions, so this rank storage is
  // always nonempty and describes the public active/resident placement set.
  private val storedSelectionSlots = selectionSlots
  // The rank never reaches `selectionSlots`: the final selected position
  // transitions directly to placement, so its index needs no extra value.
  private val rankWidth = math.max(1, log2Ceil(storedSelectionSlots))
  private def index(gid: UInt): UInt = gid(indexWidth - 1, 0)
  private def validGid(gid: UInt): Bool = gid < p.maxGaussians.U
  private def candidateRow(gid: UInt): UInt = gid.pad(p.rowWidth)(p.rowWidth - 1, 0)
  private def quantize(value: UInt): UInt = {
    val bounded = Fp32.clampUnit(value)
    // Keep the inclusive unit endpoint out of the generic float-to-integer
    // cone. The contract defines clamp(1.0) as the all-ones d field, rather
    // than allowing an implementation-specific conversion edge case.
    Mux(
      Fp32.compareGe(bounded, Fp32.One),
      255.U(8.W),
      Fp32.fp32ToUInt8(Fp32.mul(bounded, Fp32.uintToFp32(8, 255.U(8.W))))
    )
  }

  val valid = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val hotness = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(4.W))))
  val delta = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(8.W))))
  val energy = RegInit(VecInit(Seq.fill(p.maxGaussians)(Fp32.Zero)))
  val samples = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(3.W))))
  val tier = RegInit(VecInit(Seq.fill(p.maxGaussians)(ScdGsConsts.TierCandidate.U(2.W))))
  val row = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(p.rowWidth.W))))
  val touched = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val frameCount = RegInit(0.U(frameCountWidth.W))
  val phaseIdle :: phasePrepare :: phaseSelect :: phaseApply :: phaseAge :: Nil = Enum(5)
  val allocationPhase = RegInit(phaseIdle)
  val allocationIndex = RegInit(0.U(indexWidth.W))
  val selectionRank = RegInit(0.U(rankWidth.W))
  val scanBestGid = RegInit(0.U(p.gidWidth.W))
  val scanHaveBest = RegInit(false.B)
  val selected = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val rankedGid = Reg(Vec(storedSelectionSlots, UInt(p.gidWidth.W)))
  val rankedValid = RegInit(VecInit(Seq.fill(storedSelectionSlots)(false.B)))
  val activeHits = RegInit(0.U(32.W))
  val residentHits = RegInit(0.U(32.W))
  val candidateRefills = RegInit(0.U(32.W))
  val invalid = RegInit(0.U(32.W))
  // `frameEnd` is an edge input rather than a ready/valid stream. Preserve
  // every boundary observed while the sequential allocator owns the directory.
  val deferredFrameEnds = RegInit(0.U(32.W))
  val frameEndWasHigh = RegNext(io.frameEnd, false.B)
  val frameEndPulse = io.frameEnd && !frameEndWasHigh

  def existingTier(gid: UInt): UInt = Mux(valid(index(gid)), tier(index(gid)), ScdGsConsts.TierCandidate.U)
  def initializeCandidate(gid: UInt): Unit = {
    when(!valid(index(gid))) {
      valid(index(gid)) := true.B
      tier(index(gid)) := ScdGsConsts.TierCandidate.U
      row(index(gid)) := candidateRow(gid)
    }
  }
  def account(gid: UInt): Unit = {
    val t = existingTier(gid)
    hotness(index(gid)) := Mux(hotness(index(gid)) === 15.U, 15.U, hotness(index(gid)) + 1.U)
    initializeCandidate(gid)
    touched(index(gid)) := true.B
    when(t === ScdGsConsts.TierActive.U) { activeHits := activeHits + 1.U }
    when(t === ScdGsConsts.TierResident.U) { residentHits := residentHits + 1.U }
  }
  def outranks(left: UInt, right: UInt): Bool =
    hotness(index(left)) > hotness(index(right)) ||
      (hotness(index(left)) === hotness(index(right)) && delta(index(left)) > delta(index(right))) ||
      (hotness(index(left)) === hotness(index(right)) && delta(index(left)) === delta(index(right)) && left < right)

  // The allocator has three disjoint phases. Selection sees the complete,
  // unaged window. It then writes all placements before aging/resetting
  // metadata. This prevents scan order from changing an entry's rank.
  val gid = io.commitIn.bits.update.gid
  val allocating = allocationPhase =/= phaseIdle
  val propagatesLiveState = io.commitIn.bits.propagation === ScdGsConsts.PropagationCommit.U
  val producesResidency = propagatesLiveState && validGid(gid)
  io.commitIn.ready := !allocating && (!producesResidency || io.residencyOut.ready)
  val currentTier = Mux(validGid(gid), existingTier(gid), ScdGsConsts.TierCandidate.U)
  val currentRow = Mux(validGid(gid), Mux(valid(index(gid)), row(index(gid)), candidateRow(gid)), 0.U(p.rowWidth.W))
  val storedHotness = Mux(validGid(gid), hotness(index(gid)), 0.U(4.W))
  val storedDelta = Mux(validGid(gid), delta(index(gid)), 0.U(8.W))
  val storedEnergy = Mux(validGid(gid), energy(index(gid)), Fp32.Zero)
  val storedSamples = Mux(validGid(gid), samples(index(gid)), 0.U(3.W))
  val commitQuantized = quantize(io.commitIn.bits.effectiveStrength)
  // Only a state-changing commit contributes a hotness observation. A
  // confirmation retains the existing h while still updating the rest of the
  // five-frame metadata below.
  val postCommitHotness = Mux(
    propagatesLiveState,
    Mux(storedHotness === 15.U, 15.U, storedHotness + 1.U),
    storedHotness
  )
  val postCommitDelta = Mux(storedDelta > commitQuantized, storedDelta, commitQuantized)
  val postCommitEnergy = Fp32.add(storedEnergy, io.commitIn.bits.effectiveStrength)
  val postCommitSamples = Mux(storedSamples === p.siuWindow.U, storedSamples, storedSamples + 1.U)
  // Confirmation records still update FRU metadata, but their zero effective
  // strength must not enter CIU/TRU and re-expand backend work.
  // A residency transfer is the commit transfer, not a speculative view of a
  // blocked input. In particular, nothing may escape while allocation owns
  // the metadata arrays.
  io.residencyOut.valid := io.commitIn.valid && producesResidency && !allocating
  io.residencyOut.bits.propagation := io.commitIn.bits
  // StateSPM is the sole live payload owner. ScdGsTop fills these fields only
  // after the matching StateSPM response, so FRU cannot expose a stale copy.
  io.residencyOut.bits.state := 0.U.asTypeOf(new GaussianStateBundle(p))
  io.residencyOut.bits.tier := currentTier
  io.residencyOut.bits.row := currentRow
  // `candidateRefill` describes the FRU placement decision. State readiness
  // is a separate condition resolved by the top-level StateSPM handoff; an
  // active or resident lookup must not be mislabeled as a candidate merely
  // because its state response arrives one cycle later.
  io.residencyOut.bits.candidateRefill := currentTier === ScdGsConsts.TierCandidate.U
  io.residencyOut.bits.stateReady := false.B
  // Residency travels with the same commit that updates the window. CIU must
  // therefore observe the post-event E rather than waiting for a later event.
  val outgoingSamples = Mux(
    validGid(gid),
    postCommitSamples,
    0.U(3.W)
  )
  val outgoingEnergy = Mux(
    validGid(gid),
    postCommitEnergy,
    Fp32.Zero
  )
  // The five-frame window is fixed by the public contract. These are the
  // correctly rounded IEEE-754 binary32 reciprocals for counts one through
  // five, not a runtime estimate.
  val inverseSamples = MuxLookup(outgoingSamples, Fp32.Zero)(Seq(
    1.U -> "h3f800000".U(32.W),
    2.U -> "h3f000000".U(32.W),
    3.U -> "h3eaaaaab".U(32.W),
    4.U -> "h3e800000".U(32.W),
    5.U -> "h3e4ccccd".U(32.W)
  ))
  val normalizedEnergy = Fp32.mul(outgoingEnergy, inverseSamples)
  io.residencyOut.bits.energy := normalizedEnergy
  io.residencyOut.bits.sampleCount := outgoingSamples

  when(io.commitIn.fire) {
    when(validGid(gid)) {
      when(propagatesLiveState) {
        account(gid)
      }.otherwise {
        // A confirmation is a metadata observation, not a residency hit.
        // It may initialize a defensive candidate entry, but cannot raise h.
        initializeCandidate(gid)
        touched(index(gid)) := true.B
      }
      delta(index(gid)) := postCommitDelta
      // `effectiveStrength` is the canonical normalized update strength. E
      // is its mean over observed records in this five-frame window.
      energy(index(gid)) := postCommitEnergy
      samples(index(gid)) := postCommitSamples
    }.otherwise { invalid := invalid + 1.U }
  }

  io.renderReadIn.ready := !allocating && !io.commitIn.valid
  when(io.renderReadIn.fire) {
    when(validGid(io.renderReadIn.bits.gid)) { account(io.renderReadIn.bits.gid) }
      .otherwise { invalid := invalid + 1.U }
  }

  val renderGid = io.renderReadIn.bits.gid
  val renderTier = Mux(validGid(renderGid), existingTier(renderGid), ScdGsConsts.TierCandidate.U)
  val renderRow = Mux(validGid(renderGid), Mux(valid(index(renderGid)), row(index(renderGid)), candidateRow(renderGid)), 0.U(p.rowWidth.W))
  val renderHotness = Mux(validGid(renderGid), hotness(index(renderGid)), 0.U(4.W))
  val postRenderHotness = Mux(renderHotness === 15.U, 15.U, renderHotness + 1.U)
  // The allocator publishes the selected physical placement while applying it.
  // The following aging pass publishes each final post-window record, so
  // MetaSPM cannot retain a stale candidate tier after a promotion.
  val allocationTier = WireDefault(ScdGsConsts.TierCandidate.U(2.W))
  val allocationRow = Wire(UInt(p.rowWidth.W))
  allocationRow := allocationIndex
  (0 until activeSlots).foreach { slot =>
    when(rankedValid(slot) && rankedGid(slot) === allocationIndex) {
      allocationTier := ScdGsConsts.TierActive.U
      allocationRow := slot.U(p.rowWidth.W)
    }
  }
  (0 until residentSlots).foreach { offset =>
    val slot = activeSlots + offset
    when(rankedValid(slot) && rankedGid(slot) === allocationIndex) {
      allocationTier := ScdGsConsts.TierResident.U
      allocationRow := offset.U(p.rowWidth.W)
    }
  }
  val postAgeHotness = Mux(
    touched(allocationIndex),
    hotness(allocationIndex),
    hotness(allocationIndex) >> 1
  )
  io.metadataSnapshot.valid := false.B
  io.metadataSnapshot.bits := 0.U.asTypeOf(new MetaEntryBundle(p))
  io.metadataSnapshotGid := 0.U
  when(io.commitIn.fire && validGid(gid)) {
    io.metadataSnapshot.valid := true.B
    io.metadataSnapshotGid := gid
    io.metadataSnapshot.bits.tier := currentTier
    io.metadataSnapshot.bits.row := currentRow
    io.metadataSnapshot.bits.hotness := postCommitHotness
    io.metadataSnapshot.bits.delta := postCommitDelta
    io.metadataSnapshot.bits.energy := postCommitEnergy
    io.metadataSnapshot.bits.samples := postCommitSamples
    io.metadataSnapshot.bits.touched := true.B
  }.elsewhen(io.renderReadIn.fire && validGid(renderGid)) {
    io.metadataSnapshot.valid := true.B
    io.metadataSnapshotGid := renderGid
    io.metadataSnapshot.bits.tier := renderTier
    io.metadataSnapshot.bits.row := renderRow
    io.metadataSnapshot.bits.hotness := postRenderHotness
    io.metadataSnapshot.bits.delta := delta(index(renderGid))
    io.metadataSnapshot.bits.energy := energy(index(renderGid))
    io.metadataSnapshot.bits.samples := samples(index(renderGid))
    io.metadataSnapshot.bits.touched := true.B
  }.elsewhen(allocationPhase === phaseApply && valid(allocationIndex)) {
    io.metadataSnapshot.valid := true.B
    io.metadataSnapshotGid := allocationIndex
    io.metadataSnapshot.bits.tier := allocationTier
    io.metadataSnapshot.bits.row := allocationRow
    io.metadataSnapshot.bits.hotness := hotness(allocationIndex)
    io.metadataSnapshot.bits.delta := delta(allocationIndex)
    io.metadataSnapshot.bits.energy := energy(allocationIndex)
    io.metadataSnapshot.bits.samples := samples(allocationIndex)
    io.metadataSnapshot.bits.touched := touched(allocationIndex)
  }.elsewhen(allocationPhase === phaseAge && valid(allocationIndex)) {
    io.metadataSnapshot.valid := true.B
    io.metadataSnapshotGid := allocationIndex
    io.metadataSnapshot.bits.tier := tier(allocationIndex)
    io.metadataSnapshot.bits.row := row(allocationIndex)
    io.metadataSnapshot.bits.hotness := postAgeHotness
    io.metadataSnapshot.bits.delta := 0.U
    io.metadataSnapshot.bits.energy := Fp32.Zero
    io.metadataSnapshot.bits.samples := 0.U
    io.metadataSnapshot.bits.touched := false.B
  }

  when(io.candidateMissIn) { candidateRefills := candidateRefills + 1.U }

  when(frameEndPulse && allocating) {
    deferredFrameEnds := deferredFrameEnds + 1.U
  }
  val consumeDeferredFrameEnd = allocationPhase === phaseIdle && deferredFrameEnds =/= 0.U
  when(consumeDeferredFrameEnd) {
    // Retain a simultaneous new edge for the next idle cycle.
    deferredFrameEnds := Mux(frameEndPulse, deferredFrameEnds, deferredFrameEnds - 1.U)
  }
  val advanceFrameWindow = allocationPhase === phaseIdle && (frameEndPulse || consumeDeferredFrameEnd)
  when(advanceFrameWindow) {
    when(frameCount === (p.siuWindow - 1).U) {
      allocationPhase := phasePrepare
      frameCount := 0.U
    }.otherwise { frameCount := frameCount + 1.U }
  }

  when(allocationPhase === phasePrepare) {
    // Reset temporary ranking state before admitting a new window. Metadata
    // itself stays untouched until both selection and placement have ended.
    (0 until p.maxGaussians).foreach { entry => selected(entry) := false.B }
    (0 until storedSelectionSlots).foreach { slot => rankedValid(slot) := false.B }
    allocationIndex := 0.U
    selectionRank := 0.U
    scanBestGid := 0.U
    scanHaveBest := false.B
    if (selectionSlots == 0) allocationPhase := phaseApply else allocationPhase := phaseSelect
  }

  if (selectionSlots > 0) {
    when(allocationPhase === phaseSelect) {
      val scanGid = allocationIndex
      val scanCandidate = valid(scanGid) && !selected(scanGid)
      val scanWins = scanCandidate && (!scanHaveBest || outranks(scanGid, scanBestGid))
      // `finalBestGid` explicitly includes the final index of every scan
      // pass, which is the case a register-only update would otherwise lose.
      val finalBestGid = Mux(scanWins, scanGid, scanBestGid)
      val finalHaveBest = scanHaveBest || scanCandidate

      when(allocationIndex === (p.maxGaussians - 1).U) {
        when(finalHaveBest) {
          rankedGid(selectionRank) := finalBestGid
          rankedValid(selectionRank) := true.B
          selected(index(finalBestGid)) := true.B
          when(selectionRank === (selectionSlots - 1).U) {
            allocationIndex := 0.U
            allocationPhase := phaseApply
          }.otherwise {
            selectionRank := selectionRank + 1.U
            allocationIndex := 0.U
            scanBestGid := 0.U
            scanHaveBest := false.B
          }
        }.otherwise {
          allocationIndex := 0.U
          allocationPhase := phaseApply
        }
      }.otherwise {
        when(scanWins) {
          scanBestGid := scanGid
          scanHaveBest := true.B
        }
        allocationIndex := allocationIndex + 1.U
      }
    }
  }

  when(allocationPhase === phaseApply) {
    val applyGid = allocationIndex
    when(valid(applyGid)) {
      tier(applyGid) := allocationTier
      // Active and resident selected ranks map to unique partition-local rows.
      // Candidate rows are metadata only and are addressed by gid in StateSPM.
      row(applyGid) := allocationRow
    }
    when(allocationIndex === (p.maxGaussians - 1).U) {
      allocationIndex := 0.U
      allocationPhase := phaseAge
    }.otherwise { allocationIndex := allocationIndex + 1.U }
  }

  when(allocationPhase === phaseAge) {
    when(valid(allocationIndex)) {
      hotness(allocationIndex) := postAgeHotness
      delta(allocationIndex) := 0.U
      energy(allocationIndex) := Fp32.Zero
      samples(allocationIndex) := 0.U
      touched(allocationIndex) := false.B
    }
    selected(allocationIndex) := false.B
    when(allocationIndex === (p.maxGaussians - 1).U) {
      allocationIndex := 0.U
      allocationPhase := phaseIdle
    }.otherwise { allocationIndex := allocationIndex + 1.U }
  }
  io.activeHitCount := activeHits
  io.residentHitCount := residentHits
  io.candidateRefillCount := candidateRefills
  io.invalidInputCount := invalid
}
