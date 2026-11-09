package scdgs.memory

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config._

/**
  * Live Gaussian and optimizer state.
  *
  * Each valid GID has one StateSPM-owned physical copy. FRU supplies the
  * active/resident row selected by its direct-indexed tier directory. When a
  * placement changes, StateSPM installs that selected location while retaining
  * exactly one current copy. Candidate storage is deliberately GID-indexed
  * because its backing line is addressed by GID rather than by an on-chip
  * partition row.
  */
class StateSpm(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val seedIn = Flipped(Decoupled(new StateSeedBundle(p)))
    val read = Flipped(Decoupled(new StateReadReqBundle(p)))
    val readResp = Decoupled(new StateReadRespBundle(p))
    val write = Flipped(Decoupled(new StateWriteBundle(p)))
    val candidateFillIn = Flipped(Decoupled(new CandidateFillBundle(p)))
    val candidateReq = Decoupled(new MemoryLineReqBundle(p))
    val optimizerReadGid = Input(UInt(p.gidWidth.W))
    val optimizerState = Output(new OptimizerStateBundle(p))
    val optimizerValid = Output(Bool())
    val invalidInputCount = Output(UInt(32.W))
  })

  private val gidIndexWidth = math.max(1, log2Ceil(p.maxGaussians))
  private val activeIndexWidth = math.max(1, log2Ceil(p.activeCapacity))
  private val residentIndexWidth = math.max(1, log2Ceil(p.residentCapacity))

  private def index(gid: UInt): UInt = gid(gidIndexWidth - 1, 0)
  private def validGid(gid: UInt): Bool = gid < p.maxGaussians.U
  private def rowAtWidth(row: UInt, width: Int): UInt = {
    val narrowed = Wire(UInt(width.W))
    narrowed := row
    narrowed
  }
  private def activeRow(row: UInt): UInt = rowAtWidth(row, activeIndexWidth)
  private def residentRow(row: UInt): UInt = rowAtWidth(row, residentIndexWidth)
  private def activeRowValid(row: UInt): Bool = row < p.activeCapacity.U
  private def residentRowValid(row: UInt): Bool = row < p.residentCapacity.U
  private def seededRow(gid: UInt, capacity: Int): UInt = {
    val row = Wire(UInt(p.rowWidth.W))
    row := gid % capacity.U
    row
  }
  private def zeroOptimizerVector: Vec[UInt] =
    0.U.asTypeOf(Vec(p.attributeCount, UInt(p.fpWidth.W)))

  val active = Mem(p.activeCapacity, new GaussianStateBundle(p))
  val resident = Mem(p.residentCapacity, new GaussianStateBundle(p))
  val candidate = Mem(p.maxGaussians, new GaussianStateBundle(p))
  val moments = Mem(p.maxGaussians, Vec(p.attributeCount, UInt(p.fpWidth.W)))
  val variances = Mem(p.maxGaussians, Vec(p.attributeCount, UInt(p.fpWidth.W)))

  // Physical tags protect against a stale owner entry after a row is reused.
  val activeValid = RegInit(VecInit(Seq.fill(p.activeCapacity)(false.B)))
  val residentValid = RegInit(VecInit(Seq.fill(p.residentCapacity)(false.B)))
  val candidateValid = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  // Candidate payloads may exist after a seed or a physical-row displacement,
  // but only a matching refill transaction authorizes one to reach CIU.
  val candidateRefillComplete = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val activeTag = RegInit(VecInit(Seq.fill(p.activeCapacity)(0.U(p.gidWidth.W))))
  val residentTag = RegInit(VecInit(Seq.fill(p.residentCapacity)(0.U(p.gidWidth.W))))
  val candidateTag = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(p.gidWidth.W))))

  // This map identifies the one StateSPM copy TUU updates and CIU receives.
  // `ownerRow` is meaningful for active/resident only.
  val ownerValid = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val ownerTier = RegInit(VecInit(Seq.fill(p.maxGaussians)(ScdGsConsts.TierCandidate.U(2.W))))
  val ownerRow = RegInit(VecInit(Seq.fill(p.maxGaussians)(0.U(p.rowWidth.W))))

  // StateSeed does not carry an FRU row. It fills an available local row
  // first; only a full partition falls back to deterministic replacement.
  val activeFreeRows = VecInit(activeValid.map(entry => !entry)).asUInt
  val residentFreeRows = VecInit(residentValid.map(entry => !entry)).asUInt
  val activeSeedRow = Wire(UInt(p.rowWidth.W))
  val residentSeedRow = Wire(UInt(p.rowWidth.W))
  activeSeedRow := seededRow(io.seedIn.bits.state.gid, p.activeCapacity)
  residentSeedRow := seededRow(io.seedIn.bits.state.gid, p.residentCapacity)
  when(activeFreeRows.orR) { activeSeedRow := PriorityEncoder(activeFreeRows) }
  when(residentFreeRows.orR) { residentSeedRow := PriorityEncoder(residentFreeRows) }

  val respValid = RegInit(false.B)
  val resp = Reg(new StateReadRespBundle(p))
  val requestValid = RegInit(false.B)
  val request = Reg(new MemoryLineReqBundle(p))
  val missPending = RegInit(false.B)
  val missRequest = Reg(new StateReadReqBundle(p))
  val invalid = RegInit(0.U(32.W))

  /** Invalidate the current physical copy, but never a reused row. */
  def invalidateOwnedCopy(gid: UInt): Unit = {
    val gidIndex = index(gid)
    switch(ownerTier(gidIndex)) {
      is(ScdGsConsts.TierActive.U) {
        when(activeRowValid(ownerRow(gidIndex))) {
          val row = activeRow(ownerRow(gidIndex))
          when(activeValid(row) && activeTag(row) === gid) { activeValid(row) := false.B }
        }
      }
      is(ScdGsConsts.TierResident.U) {
        when(residentRowValid(ownerRow(gidIndex))) {
          val row = residentRow(ownerRow(gidIndex))
          when(residentValid(row) && residentTag(row) === gid) { residentValid(row) := false.B }
        }
      }
      is(ScdGsConsts.TierCandidate.U) {
        when(candidateValid(gidIndex) && candidateTag(gidIndex) === gid) { candidateValid(gidIndex) := false.B }
        candidateRefillComplete(gidIndex) := false.B
      }
    }
  }

  /**
    * Install one current physical copy. Callers validate
    * active/resident rows before invoking this helper.
    */
  def installOwner(
      gid: UInt,
      payload: GaussianStateBundle,
      targetTier: UInt,
      targetRow: UInt,
      resetOptimizer: Bool,
      candidateFillComplete: Bool
  ): Unit = {
    val gidIndex = index(gid)
    val sameLocation = ownerValid(gidIndex) && ownerTier(gidIndex) === targetTier &&
      (targetTier === ScdGsConsts.TierCandidate.U || ownerRow(gidIndex) === targetRow)

    when(ownerValid(gidIndex) && !sameLocation) { invalidateOwnedCopy(gid) }
    switch(targetTier) {
      is(ScdGsConsts.TierActive.U) {
        val row = activeRow(targetRow)
        val displacedGid = activeTag(row)
        val displacedIndex = index(displacedGid)
        val displacedOwner = activeValid(row) && activeTag(row) =/= gid &&
          validGid(displacedGid) && ownerValid(displacedIndex) &&
          ownerTier(displacedIndex) === ScdGsConsts.TierActive.U &&
          ownerRow(displacedIndex) === targetRow
        when(displacedOwner) {
          // FRU can reassign a physical row before the displaced Gaussian is
          // next selected. Preserve that live state in its GID-indexed
          // candidate partition, so its later FRU-directed migration cannot
          // observe a lost active copy.
          candidate.write(displacedIndex, active.read(row))
          candidateValid(displacedIndex) := true.B
          candidateTag(displacedIndex) := displacedGid
          candidateRefillComplete(displacedIndex) := false.B
          ownerValid(displacedIndex) := true.B
          ownerTier(displacedIndex) := ScdGsConsts.TierCandidate.U
          ownerRow(displacedIndex) := 0.U
        }
        active.write(row, payload)
        activeValid(row) := true.B
        activeTag(row) := gid
      }
      is(ScdGsConsts.TierResident.U) {
        val row = residentRow(targetRow)
        val displacedGid = residentTag(row)
        val displacedIndex = index(displacedGid)
        val displacedOwner = residentValid(row) && residentTag(row) =/= gid &&
          validGid(displacedGid) && ownerValid(displacedIndex) &&
          ownerTier(displacedIndex) === ScdGsConsts.TierResident.U &&
          ownerRow(displacedIndex) === targetRow
        when(displacedOwner) {
          // Keep the old resident payload reachable until its next explicit
          // placement request, including a promotion that reuses this row.
          candidate.write(displacedIndex, resident.read(row))
          candidateValid(displacedIndex) := true.B
          candidateTag(displacedIndex) := displacedGid
          candidateRefillComplete(displacedIndex) := false.B
          ownerValid(displacedIndex) := true.B
          ownerTier(displacedIndex) := ScdGsConsts.TierCandidate.U
          ownerRow(displacedIndex) := 0.U
        }
        resident.write(row, payload)
        residentValid(row) := true.B
        residentTag(row) := gid
      }
      is(ScdGsConsts.TierCandidate.U) {
        candidate.write(gidIndex, payload)
        candidateValid(gidIndex) := true.B
        candidateTag(gidIndex) := gid
        candidateRefillComplete(gidIndex) := candidateFillComplete
      }
    }
    when(targetTier =/= ScdGsConsts.TierCandidate.U) { candidateRefillComplete(gidIndex) := false.B }
    ownerValid(gidIndex) := true.B
    ownerTier(gidIndex) := targetTier
    ownerRow(gidIndex) := Mux(targetTier === ScdGsConsts.TierCandidate.U, 0.U, targetRow)
    when(resetOptimizer) {
      moments.write(gidIndex, zeroOptimizerVector)
      variances.write(gidIndex, zeroOptimizerVector)
    }
  }

  val readGid = io.read.bits.gid
  val readIndex = index(readGid)
  val readOwnerKnown = validGid(readGid) && ownerValid(readIndex)
  val readOwnerActiveRow = activeRow(ownerRow(readIndex))
  val readOwnerResidentRow = residentRow(ownerRow(readIndex))
  val readActiveState = active.read(readOwnerActiveRow)
  val readResidentState = resident.read(readOwnerResidentRow)
  val readCandidateState = candidate.read(readIndex)
  val readActiveHit = readOwnerKnown && ownerTier(readIndex) === ScdGsConsts.TierActive.U &&
    activeRowValid(ownerRow(readIndex)) && activeValid(readOwnerActiveRow) &&
    activeTag(readOwnerActiveRow) === readGid && readActiveState.gid === readGid
  val readResidentHit = readOwnerKnown && ownerTier(readIndex) === ScdGsConsts.TierResident.U &&
    residentRowValid(ownerRow(readIndex)) && residentValid(readOwnerResidentRow) &&
    residentTag(readOwnerResidentRow) === readGid && readResidentState.gid === readGid
  val readCandidateHit = readOwnerKnown && ownerTier(readIndex) === ScdGsConsts.TierCandidate.U &&
    candidateValid(readIndex) && candidateTag(readIndex) === readGid && readCandidateState.gid === readGid
  val readCandidateComplete = readCandidateHit && candidateRefillComplete(readIndex)
  val readStorageHit = readActiveHit || readResidentHit || readCandidateComplete
  val readOwnedState = Wire(new GaussianStateBundle(p))
  readOwnedState := Mux(readActiveHit, readActiveState,
    Mux(readResidentHit, readResidentState,
      Mux(readCandidateComplete, readCandidateState, 0.U.asTypeOf(new GaussianStateBundle(p)))))

  val readTierKnown = io.read.bits.tier === ScdGsConsts.TierActive.U ||
    io.read.bits.tier === ScdGsConsts.TierResident.U ||
    io.read.bits.tier === ScdGsConsts.TierCandidate.U
  val readTargetRowValid = Mux(io.read.bits.tier === ScdGsConsts.TierActive.U,
    activeRowValid(io.read.bits.row),
    Mux(io.read.bits.tier === ScdGsConsts.TierResident.U,
      residentRowValid(io.read.bits.row), true.B))
  val readTargetMatchesOwner = readOwnerKnown && ownerTier(readIndex) === io.read.bits.tier &&
    (io.read.bits.tier === ScdGsConsts.TierCandidate.U || ownerRow(readIndex) === io.read.bits.row)
  // The paper requires candidate state to complete an explicit refill before
  // CIU can observe it. A cached payload is usable only after that transaction.
  val unresolvedCandidateOwner = readCandidateHit && !candidateRefillComplete(readIndex)
  val candidatePlacementNeedsRefill = io.read.bits.tier === ScdGsConsts.TierCandidate.U && !readCandidateComplete
  val requiresCandidateRefill = unresolvedCandidateOwner || candidatePlacementNeedsRefill

  // A fill is accepted only for a cold transaction whose request was already
  // accepted. `bytes = lineBytes` describes request alignment/granularity;
  // CandidateFillBundle restores the complete StateSPM-owned persistent state.
  val fillCanAccept = missPending && !requestValid && !respValid &&
    validGid(io.candidateFillIn.bits.state.gid) &&
    io.candidateFillIn.bits.state.gid === missRequest.gid
  val fillWillFire = io.candidateFillIn.valid && fillCanAccept
  io.candidateFillIn.ready := fillCanAccept

  // One memory-update source is accepted per cycle. This avoids ambiguous
  // same-row writes while preserving independent request and response links.
  io.write.ready := !fillWillFire
  val writeWillFire = io.write.valid && io.write.ready
  io.seedIn.ready := !fillWillFire && !writeWillFire
  val seedWillFire = io.seedIn.valid && io.seedIn.ready
  io.read.ready := !respValid && !missPending && !requestValid &&
    !fillWillFire && !writeWillFire && !seedWillFire

  when(io.seedIn.fire) {
    val gid = io.seedIn.bits.state.gid
    val tier = io.seedIn.bits.tier
    val tierKnown = tier === ScdGsConsts.TierActive.U || tier === ScdGsConsts.TierResident.U ||
      tier === ScdGsConsts.TierCandidate.U
    val row = Wire(UInt(p.rowWidth.W))
    row := 0.U
    when(tier === ScdGsConsts.TierActive.U) { row := activeSeedRow }
      .elsewhen(tier === ScdGsConsts.TierResident.U) { row := residentSeedRow }
    // A StateSeed initializes one Gaussian before the live loop begins. Once
    // a StateSPM owner exists, only TUU or a matching refill may change it.
    when(validGid(gid) && tierKnown && !ownerValid(index(gid))) {
      installOwner(gid, io.seedIn.bits.state, tier, row, resetOptimizer = true.B, candidateFillComplete = false.B)
    }.otherwise { invalid := invalid + 1.U }
  }

  val writeGid = io.write.bits.gid
  val writeIndex = index(writeGid)
  val writeOwnerKnown = validGid(writeGid) && ownerValid(writeIndex)
  val writeActiveRow = activeRow(ownerRow(writeIndex))
  val writeResidentRow = residentRow(ownerRow(writeIndex))
  val writeActiveHit = writeOwnerKnown && ownerTier(writeIndex) === ScdGsConsts.TierActive.U &&
    activeRowValid(ownerRow(writeIndex)) && activeValid(writeActiveRow) && activeTag(writeActiveRow) === writeGid
  val writeResidentHit = writeOwnerKnown && ownerTier(writeIndex) === ScdGsConsts.TierResident.U &&
    residentRowValid(ownerRow(writeIndex)) && residentValid(writeResidentRow) && residentTag(writeResidentRow) === writeGid
  val writeCandidateHit = writeOwnerKnown && ownerTier(writeIndex) === ScdGsConsts.TierCandidate.U &&
    candidateValid(writeIndex) && candidateTag(writeIndex) === writeGid && candidateRefillComplete(writeIndex)
  val writeOwnerHit = writeActiveHit || writeResidentHit || writeCandidateHit

  when(io.write.fire) {
    when(validGid(writeGid) && io.write.bits.state.gid === writeGid && writeOwnerHit) {
      when(writeActiveHit) { active.write(writeActiveRow, io.write.bits.state) }
        .elsewhen(writeResidentHit) { resident.write(writeResidentRow, io.write.bits.state) }
        .otherwise { candidate.write(writeIndex, io.write.bits.state) }
      moments.write(writeIndex, io.write.bits.adamMoment)
      variances.write(writeIndex, io.write.bits.adamVariance)
    }.otherwise {
      when(validGid(writeGid) && writeOwnerKnown && !writeOwnerHit) {
        invalidateOwnedCopy(writeGid)
        ownerValid(writeIndex) := false.B
      }
      invalid := invalid + 1.U
    }
  }

  when(io.read.fire) {
    when(!validGid(readGid)) {
      resp := 0.U.asTypeOf(new StateReadRespBundle(p))
      resp.state.gid := readGid
      resp.frameId := io.read.bits.frameId
      resp.hit := false.B
      respValid := true.B
      invalid := invalid + 1.U
    }.elsewhen(!readTierKnown || !readTargetRowValid) {
      resp := 0.U.asTypeOf(new StateReadRespBundle(p))
      resp.state.gid := readGid
      resp.frameId := io.read.bits.frameId
      resp.hit := false.B
      respValid := true.B
      invalid := invalid + 1.U
    }.elsewhen(readStorageHit && !requiresCandidateRefill) {
      resp.state := readOwnedState
      resp.frameId := io.read.bits.frameId
      resp.hit := true.B
      respValid := true.B
      when(!readTargetMatchesOwner) {
        installOwner(
          readGid,
          readOwnedState,
          io.read.bits.tier,
          io.read.bits.row,
          resetOptimizer = false.B,
          candidateFillComplete = false.B
        )
      }
    }.otherwise {
      // A candidate placement, or a candidate payload that has not completed
      // its refill, remains pending. The complete fill is installed at the
      // placement that caused the lookup before CIU can receive a response.
      when(readOwnerKnown) {
        invalidateOwnedCopy(readGid)
        ownerValid(readIndex) := false.B
      }
      request.address := readGid * p.lineBytes.U
      request.gid := readGid
      request.frameId := io.read.bits.frameId
      request.bytes := p.lineBytes.U
      requestValid := true.B
      missRequest := io.read.bits
      missPending := true.B
    }
  }

  when(io.candidateFillIn.fire) {
    val gid = io.candidateFillIn.bits.state.gid
    installOwner(
      gid,
      io.candidateFillIn.bits.state,
      missRequest.tier,
      missRequest.row,
      resetOptimizer = false.B,
      candidateFillComplete = true.B
    )
    moments.write(index(gid), io.candidateFillIn.bits.adamMoment)
    variances.write(index(gid), io.candidateFillIn.bits.adamVariance)
    resp.state := io.candidateFillIn.bits.state
    resp.frameId := missRequest.frameId
    resp.hit := true.B
    respValid := true.B
    missPending := false.B
  }

  io.readResp.valid := respValid
  io.readResp.bits := resp
  when(io.readResp.fire) { respValid := false.B }
  io.candidateReq.valid := requestValid
  io.candidateReq.bits := request
  when(io.candidateReq.fire) { requestValid := false.B }

  val optimizerIndex = index(io.optimizerReadGid)
  val optimizerOwnerKnown = validGid(io.optimizerReadGid) && ownerValid(optimizerIndex)
  val optimizerActiveRow = activeRow(ownerRow(optimizerIndex))
  val optimizerResidentRow = residentRow(ownerRow(optimizerIndex))
  val optimizerActiveState = active.read(optimizerActiveRow)
  val optimizerResidentState = resident.read(optimizerResidentRow)
  val optimizerCandidateState = candidate.read(optimizerIndex)
  val optimizerActiveHit = optimizerOwnerKnown && ownerTier(optimizerIndex) === ScdGsConsts.TierActive.U &&
    activeRowValid(ownerRow(optimizerIndex)) && activeValid(optimizerActiveRow) &&
    activeTag(optimizerActiveRow) === io.optimizerReadGid && optimizerActiveState.gid === io.optimizerReadGid
  val optimizerResidentHit = optimizerOwnerKnown && ownerTier(optimizerIndex) === ScdGsConsts.TierResident.U &&
    residentRowValid(ownerRow(optimizerIndex)) && residentValid(optimizerResidentRow) &&
    residentTag(optimizerResidentRow) === io.optimizerReadGid && optimizerResidentState.gid === io.optimizerReadGid
  val optimizerCandidateHit = optimizerOwnerKnown && ownerTier(optimizerIndex) === ScdGsConsts.TierCandidate.U &&
    candidateValid(optimizerIndex) && candidateTag(optimizerIndex) === io.optimizerReadGid &&
    optimizerCandidateState.gid === io.optimizerReadGid && candidateRefillComplete(optimizerIndex)
  val optimizerHasState = optimizerActiveHit || optimizerResidentHit || optimizerCandidateHit

  io.optimizerState.gid := io.optimizerReadGid
  io.optimizerState.state := Mux(optimizerActiveHit, optimizerActiveState,
    Mux(optimizerResidentHit, optimizerResidentState,
      Mux(optimizerCandidateHit, optimizerCandidateState, 0.U.asTypeOf(new GaussianStateBundle(p)))))
  io.optimizerState.adamMoment := Mux(optimizerHasState, moments.read(optimizerIndex), zeroOptimizerVector)
  io.optimizerState.adamVariance := Mux(optimizerHasState, variances.read(optimizerIndex), zeroOptimizerVector)
  io.optimizerValid := optimizerHasState
  io.invalidInputCount := invalid
}
