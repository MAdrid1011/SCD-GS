package scdgs.config

/**
  * Public reference-microarchitecture parameters.
  *
  * These values define executable state and interface shapes. They are not a
  * claim about a paper PPA configuration.
  */
case class ScdGsParams(
  maxGaussians: Int = 4096,
  gidWidth: Int = 12,
  frameWidth: Int = 16,
  fpWidth: Int = 32,
  tileWidth: Int = 12,
  rowWidth: Int = 12,
  activeCapacity: Int = 170,
  residentCapacity: Int = 512,
  tilePixels: Int = 256,
  shDegree: Int = 3,
  siuWindow: Int = 5,
  // Paper defaults: tauPhi=0.16, tauRho=0.90, tauD=0.15.
  tauPhi: BigInt = BigInt("3e23d70a", 16),
  tauRho: BigInt = BigInt("3f666666", 16),
  tauD: BigInt = BigInt("3e19999a", 16),
  transmittanceStop: BigInt = BigInt("3d4ccccd", 16),
  lineBytes: Int = 64,
  renderQueueDepth: Int = 64,
  trainingLanes: Int = 16,
  adamBeta1: BigInt = BigInt("3f666666", 16),
  adamBeta2: BigInt = BigInt("3f7fbe77", 16),
  adamLearningRate: BigInt = BigInt("3a83126f", 16),
  adamEpsilon: BigInt = BigInt("322bcc77", 16)
) {
  val shBases: Int = (shDegree + 1) * (shDegree + 1)
  val shCoefficients: Int = 3 * shBases
  val attributeCount: Int = 3 + 6 + 1 + shCoefficients
  val maxFrameId: BigInt = (BigInt(1) << frameWidth) - 1

  require(maxGaussians > 0)
  require(gidWidth > 0 && BigInt(maxGaussians) <= (BigInt(1) << gidWidth), "gidWidth must represent every Gaussian ID")
  require(frameWidth > 0, "frameWidth must be positive")
  require(fpWidth == 32, "the public mechanism uses IEEE-754 binary32 payloads")
  require(tileWidth > 0 && tileWidth < 32, "tileWidth must form a nonempty 32-bit pixel field")
  require(
    rowWidth > 0 &&
      BigInt(maxGaussians) <= (BigInt(1) << rowWidth) &&
      BigInt(activeCapacity) <= (BigInt(1) << rowWidth) &&
      BigInt(residentCapacity) <= (BigInt(1) << rowWidth),
    "rowWidth must represent every candidate GID and physical partition row"
  )
  require(activeCapacity > 0 && residentCapacity > 0, "StateSPM requires active and resident partitions")
  require(tilePixels > 0 && tilePixels < (1 << 16), "tilePixels must fit the public 16-bit conversion path")
  require(shDegree == 3, "the public mechanism fixes degree-3 SH")
  require(siuWindow == 5, "the public mechanism fixes five SIU observations and five FRU frames")
  require(lineBytes == 64, "candidate refill requests use 64-byte lines")
  require(renderQueueDepth > 0, "TRU requires a nonempty frame buffer")
  require(trainingLanes == 16, "the public mechanism fixes sixteen logical training lanes")
}

object ScdGsConsts {
  val StructuralNone = 0
  val StructuralInsert = 1
  val StructuralDelete = 2
  val StructuralInvalidate = 3
  val StructuralReset = 4

  val TierActive = 0
  val TierResident = 1
  val TierCandidate = 2

  val RequestUser = 0
  val RequestTraining = 1

  val PropagationCommit = 0
  val PropagationConfirmation = 1

  val PathFull = 0
  val PathLight = 1
  val PathFilter = 2
}
