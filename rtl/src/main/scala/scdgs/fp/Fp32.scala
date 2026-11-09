package scdgs.fp

import chisel3._
import chisel3.util.Cat
import hardfloat._

object Fp32 {
  val Zero: UInt = "h00000000".U(32.W)
  val One: UInt = "h3f800000".U(32.W)
  val Half: UInt = "h3f000000".U(32.W)
  val Quarter: UInt = "h3e800000".U(32.W)
  val Eighth: UInt = "h3e000000".U(32.W)
  val Two: UInt = "h40000000".U(32.W)
  val Four: UInt = "h40800000".U(32.W)
  val Inv255: UInt = "h3b808081".U(32.W)

  private def roundNearEven: UInt = hardfloat.consts.round_near_even

  private def tininessAfter: UInt = hardfloat.consts.tininess_afterRounding

  private def toRec32(x: UInt): UInt = recFNFromFN(8, 24, x).asUInt

  private def fromRec32(x: Bits): UInt = fNFromRecFN(8, 24, x).asUInt

  def negate(x: UInt): UInt = Cat(~x(31), x(30, 0)).asUInt

  def abs(x: UInt): UInt = Mux(x(31), negate(x), x)

  def fma(a: UInt, b: UInt, c: UInt): UInt = {
    val unit = Module(new MulAddRecFN(8, 24))
    unit.io.op := 0.U
    unit.io.a := toRec32(a)
    unit.io.b := toRec32(b)
    unit.io.c := toRec32(c)
    unit.io.roundingMode := roundNearEven
    unit.io.detectTininess := tininessAfter
    fromRec32(unit.io.out)
  }

  def add(a: UInt, b: UInt): UInt = fma(a, One, b)

  def sub(a: UInt, b: UInt): UInt = add(a, negate(b))

  def mul(a: UInt, b: UInt): UInt = fma(a, b, Zero)

  def compareLt(a: UInt, b: UInt): Bool = {
    val unit = Module(new CompareRecFN(8, 24))
    unit.io.a := toRec32(a)
    unit.io.b := toRec32(b)
    unit.io.signaling := false.B
    unit.io.lt
  }

  /** IEEE-754 numerical equality, including the equivalence of +0 and -0. */
  def compareEq(a: UInt, b: UInt): Bool = {
    val unit = Module(new CompareRecFN(8, 24))
    unit.io.a := toRec32(a)
    unit.io.b := toRec32(b)
    unit.io.signaling := false.B
    unit.io.eq
  }

  /** Ordered IEEE-754 greater-than-or-equal comparison. */
  def compareGe(a: UInt, b: UInt): Bool = {
    val unit = Module(new CompareRecFN(8, 24))
    unit.io.a := toRec32(a)
    unit.io.b := toRec32(b)
    unit.io.signaling := false.B
    unit.io.gt || unit.io.eq
  }

  def clampUnit(x: UInt): UInt = Mux(compareLt(x, One), Mux(compareLt(x, Zero), Zero, x), One)

  /** Valid denominator encoding for the public SIU projected-radius formula. */
  def isPositiveNormal(x: UInt): Bool = {
    val exp = x(30, 23)
    exp =/= 0.U && exp =/= "hff".U && !x(31)
  }

  /** IEEE-754 binary32 values with a defined numerical ordering. */
  def isFinite(x: UInt): Bool = x(30, 23) =/= "hff".U

  def fp16ToFp32(x: UInt): UInt = {
    val unit = Module(new RecFNToRecFN(5, 11, 8, 24))
    unit.io.in := recFNFromFN(5, 11, x)
    unit.io.roundingMode := roundNearEven
    unit.io.detectTininess := tininessAfter
    fromRec32(unit.io.out)
  }

  /** Packs a binary32 value for the light SH path. */
  def fp32ToFp16(x: UInt): UInt = {
    val unit = Module(new RecFNToRecFN(8, 24, 5, 11))
    unit.io.in := toRec32(x)
    unit.io.roundingMode := roundNearEven
    unit.io.detectTininess := tininessAfter
    fNFromRecFN(5, 11, unit.io.out).asUInt
  }

  def uintToFp32(width: Int, x: UInt): UInt = {
    val unit = Module(new INToRecFN(width, 8, 24))
    unit.io.signedIn := false.B
    unit.io.in := x
    unit.io.roundingMode := roundNearEven
    unit.io.detectTininess := tininessAfter
    fromRec32(unit.io.out)
  }

  def fp32ToUInt(width: Int, x: UInt): UInt = {
    val unit = Module(new RecFNToIN(8, 24, width))
    unit.io.in := toRec32(x)
    unit.io.roundingMode := hardfloat.consts.round_minMag
    unit.io.signedOut := false.B
    unit.io.out.asUInt
  }

  def fp32ToUInt8(x: UInt): UInt = fp32ToUInt(8, x)

  def byteFrac(x: UInt): UInt = mul(uintToFp32(8, x), Inv255)
}
