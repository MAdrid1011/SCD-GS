package scdgs.config

import org.scalatest.funsuite.AnyFunSuite

/** Keeps the public candidate-row encoding wide enough for every legal GID. */
class ScdGsParamsSpec extends AnyFunSuite {
  test("row and GID fields represent the complete public Gaussian domain") {
    val defaults = ScdGsParams()
    assert(defaults.maxGaussians <= (BigInt(1) << defaults.gidWidth))
    assert(defaults.maxGaussians <= (BigInt(1) << defaults.rowWidth))
    assertThrows[IllegalArgumentException] {
      ScdGsParams(maxGaussians = 4096, rowWidth = 10)
    }
  }

  test("rejects parameterizations outside the public mechanism contract") {
    Seq(
      () => ScdGsParams(frameWidth = 0),
      () => ScdGsParams(fpWidth = 16),
      () => ScdGsParams(tileWidth = 0),
      () => ScdGsParams(activeCapacity = 0),
      () => ScdGsParams(residentCapacity = 0),
      () => ScdGsParams(tilePixels = 0),
      () => ScdGsParams(shDegree = 2),
      () => ScdGsParams(siuWindow = 4),
      () => ScdGsParams(lineBytes = 32),
      () => ScdGsParams(renderQueueDepth = 0),
      () => ScdGsParams(trainingLanes = 8)
    ).foreach { makeInvalid =>
      assertThrows[IllegalArgumentException](makeInvalid())
    }
  }
}
