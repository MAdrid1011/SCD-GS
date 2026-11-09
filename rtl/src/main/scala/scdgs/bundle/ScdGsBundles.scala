package scdgs.bundle

import chisel3._
import scdgs.config.ScdGsParams

/** Full live Gaussian state stored by StateSPM and consumed by CIU/TRU. */
class GaussianStateBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val mean = Vec(3, UInt(p.fpWidth.W))
  val covariance = Vec(6, UInt(p.fpWidth.W))
  val opacity = UInt(p.fpWidth.W)
  val sh = Vec(p.shCoefficients, UInt(p.fpWidth.W))
}

/** The update record produced by TUU and consumed by SIU. */
class CanonicalUpdateBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val frameId = UInt(p.frameWidth.W)
  val deltaMu = Vec(3, UInt(p.fpWidth.W))
  val deltaSigma = Vec(6, UInt(p.fpWidth.W))
  val deltaOpacity = UInt(p.fpWidth.W)
  val deltaSh = Vec(p.shCoefficients, UInt(p.fpWidth.W))
  val strength = UInt(p.fpWidth.W)
  val structural = UInt(3.W)
}

/**
 * Full projection and view record. SplatSPM retains a training instance for
 * SIU; CIU separately receives the current instance for the render request.
 */
class ProjectionSummaryBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val frameId = UInt(p.frameWidth.W)
  val rCam = Vec(3, Vec(3, UInt(p.fpWidth.W)))
  val jPi = Vec(2, Vec(3, UInt(p.fpWidth.W)))
  val jCov = Vec(2, Vec(3, UInt(p.fpWidth.W)))
  val viewDirection = Vec(3, UInt(p.fpWidth.W))
  val shBasis = Vec(p.shBases, UInt(p.fpWidth.W))
  val center = Vec(2, UInt(p.fpWidth.W))
  val conic = Vec(3, UInt(p.fpWidth.W))
  val radius = UInt(p.fpWidth.W)
  val area = UInt(p.fpWidth.W)
  val opacity = UInt(p.fpWidth.W)
  val viewColor = Vec(3, UInt(p.fpWidth.W))
  val requestKind = UInt(1.W)
  val tile = UInt(p.tileWidth.W)
  val depth = UInt(p.fpWidth.W)
}

class PropagationBundle(p: ScdGsParams) extends Bundle {
  val update = new CanonicalUpdateBundle(p)
  val projection = new ProjectionSummaryBundle(p)
  val psi = UInt(p.fpWidth.W)
  val phi = UInt(p.fpWidth.W)
  val effectiveStrength = UInt(p.fpWidth.W)
  val propagation = UInt(1.W)
}

class RenderReadBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val frameId = UInt(p.frameWidth.W)
}

class ResidencyBundle(p: ScdGsParams) extends Bundle {
  val propagation = new PropagationBundle(p)
  val state = new GaussianStateBundle(p)
  val tier = UInt(2.W)
  val row = UInt(p.rowWidth.W)
  val candidateRefill = Bool()
  // CIU must keep a request conservative until StateSPM has supplied a state.
  val stateReady = Bool()
  val energy = UInt(p.fpWidth.W)
  val sampleCount = UInt(3.W)
}

class IssueBundle(p: ScdGsParams) extends Bundle {
  val residency = new ResidencyBundle(p)
  // The current render view is distinct from the prior training summary held
  // in `residency.propagation.projection` for the next SIU decision.
  val view = new ProjectionSummaryBundle(p)
  val path = UInt(2.W)
  val supportArea = UInt(p.fpWidth.W)
  val colorEstimate = Vec(3, UInt(p.fpWidth.W))
  val renderingPotential = UInt(p.fpWidth.W)
  val kappa = UInt(p.fpWidth.W)
  val rho = UInt(p.fpWidth.W)
  val omega = UInt(p.fpWidth.W)
}

class RenderTraceBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val frameId = UInt(p.frameWidth.W)
  val pixel = UInt(32.W)
  val blendOrder = UInt(16.W)
  val coverage = UInt(p.fpWidth.W)
  val transmittance = UInt(p.fpWidth.W)
  val color = Vec(3, UInt(p.fpWidth.W))
  val projection = new ProjectionSummaryBundle(p)
  val path = UInt(2.W)
}

class TargetPixelBundle(p: ScdGsParams) extends Bundle {
  val pixel = UInt(32.W)
  val frameId = UInt(p.frameWidth.W)
  val rgb = Vec(3, UInt(p.fpWidth.W))
}

/** Rendered representative-pixel record retained in FrameSPM alongside the live dataflow. */
class PixelResultBundle(p: ScdGsParams) extends Bundle {
  val pixel = UInt(32.W)
  val frameId = UInt(p.frameWidth.W)
  val rgb = Vec(3, UInt(p.fpWidth.W))
  val transmittance = UInt(p.fpWidth.W)
}

class StateSeedBundle(p: ScdGsParams) extends Bundle {
  val state = new GaussianStateBundle(p)
  val tier = UInt(2.W)
}

class StateReadReqBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val frameId = UInt(p.frameWidth.W)
  val tier = UInt(2.W)
  // FRU owns active/resident placement. Candidate storage is GID-indexed.
  val row = UInt(p.rowWidth.W)
}

class StateReadRespBundle(p: ScdGsParams) extends Bundle {
  val state = new GaussianStateBundle(p)
  val frameId = UInt(p.frameWidth.W)
  val hit = Bool()
}

class OptimizerStateBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val state = new GaussianStateBundle(p)
  val adamMoment = Vec(p.attributeCount, UInt(p.fpWidth.W))
  val adamVariance = Vec(p.attributeCount, UInt(p.fpWidth.W))
}

class StateWriteBundle(p: ScdGsParams) extends Bundle {
  val gid = UInt(p.gidWidth.W)
  val state = new GaussianStateBundle(p)
  val adamMoment = Vec(p.attributeCount, UInt(p.fpWidth.W))
  val adamVariance = Vec(p.attributeCount, UInt(p.fpWidth.W))
}

class CandidateFillBundle(p: ScdGsParams) extends Bundle {
  val state = new GaussianStateBundle(p)
  // A refill must restore every StateSPM-owned persistent field for the GID.
  // The request remains a 64-byte line-granularity contract; this public
  // module-boundary record is not a literal memory-line payload.
  val adamMoment = Vec(p.attributeCount, UInt(p.fpWidth.W))
  val adamVariance = Vec(p.attributeCount, UInt(p.fpWidth.W))
}

class MemoryLineReqBundle(p: ScdGsParams) extends Bundle {
  val address = UInt(64.W)
  val gid = UInt(p.gidWidth.W)
  val frameId = UInt(p.frameWidth.W)
  val bytes = UInt(7.W)
}

/** Published snapshot of FRU residency metadata. */
class MetaEntryBundle(p: ScdGsParams) extends Bundle {
  val tier = UInt(2.W)
  val row = UInt(p.rowWidth.W)
  val hotness = UInt(4.W)
  val delta = UInt(8.W)
  val energy = UInt(p.fpWidth.W)
  val samples = UInt(3.W)
  val touched = Bool()
}

class MechanismCountersBundle extends Bundle {
  val siuCommit = UInt(32.W)
  val siuConfirmation = UInt(32.W)
  val fruActiveHit = UInt(32.W)
  val fruResidentHit = UInt(32.W)
  val fruCandidateRefill = UInt(32.W)
  val ciuFull = UInt(32.W)
  val ciuLight = UInt(32.W)
  val ciuFilter = UInt(32.W)
  val truEarlyStop = UInt(32.W)
  val truShOps = UInt(32.W)
  val truCoverageOps = UInt(32.W)
  val truBlendOps = UInt(32.W)
  val tuuUpdates = UInt(32.W)
  val queueStalls = UInt(32.W)
  // Includes rejected identifiers and rejected cross-record transactions.
  val invalidInput = UInt(32.W)
}
