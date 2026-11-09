package scdgs.math

import chisel3._
import chisel3.util._
import hardfloat._

/** One-at-a-time IEEE-754 binary32 division wrapper. */
class DivF32 extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Bundle {
      val numerator = UInt(32.W)
      val denominator = UInt(32.W)
    }))
    val out = Decoupled(UInt(32.W))
  })

  val unit = Module(new DivSqrtRecFN_small(8, 24, 0))
  val busy = RegInit(false.B)
  val resultValid = RegInit(false.B)
  val result = RegInit(0.U(32.W))

  unit.io.sqrtOp := false.B
  unit.io.a := recFNFromFN(8, 24, io.in.bits.numerator)
  unit.io.b := recFNFromFN(8, 24, io.in.bits.denominator)
  unit.io.roundingMode := hardfloat.consts.round_near_even
  unit.io.detectTininess := hardfloat.consts.tininess_afterRounding
  unit.io.inValid := io.in.valid && !busy && !resultValid && unit.io.inReady
  io.in.ready := !busy && !resultValid && unit.io.inReady
  when(io.in.fire) { busy := true.B }
  when(unit.io.outValid_div) {
    result := fNFromRecFN(8, 24, unit.io.out)
    resultValid := true.B
    busy := false.B
  }
  io.out.valid := resultValid
  io.out.bits := result
  when(io.out.fire) { resultValid := false.B }
}

/** One-at-a-time IEEE-754 binary32 square-root wrapper for SIU covariance impact. */
class SqrtF32 extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(32.W)))
    val out = Decoupled(UInt(32.W))
  })

  val unit = Module(new DivSqrtRecFN_small(8, 24, 0))
  val busy = RegInit(false.B)
  val resultValid = RegInit(false.B)
  val result = RegInit(0.U(32.W))

  unit.io.sqrtOp := true.B
  unit.io.a := recFNFromFN(8, 24, io.in.bits)
  unit.io.b := 0.U
  unit.io.roundingMode := hardfloat.consts.round_near_even
  unit.io.detectTininess := hardfloat.consts.tininess_afterRounding
  unit.io.inValid := io.in.valid && !busy && !resultValid && unit.io.inReady
  io.in.ready := !busy && !resultValid && unit.io.inReady
  when(io.in.fire) { busy := true.B }
  when(unit.io.outValid_sqrt) {
    result := fNFromRecFN(8, 24, unit.io.out)
    resultValid := true.B
    busy := false.B
  }
  io.out.valid := resultValid
  io.out.bits := result
  when(io.out.fire) { resultValid := false.B }
}
