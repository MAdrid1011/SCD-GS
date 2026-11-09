package scdgs.fp

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import scdgs.TestSupport._
import scdgs.math.DivF32
import org.scalatest.funsuite.AnyFunSuite

private class Fp32Harness extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val sum = Output(UInt(32.W))
    val product = Output(UInt(32.W))
    val clamped = Output(UInt(32.W))
    val equal = Output(Bool())
    val greaterOrEqual = Output(Bool())
    val finite = Output(Bool())
  })
  io.sum := Fp32.add(io.a, io.b)
  io.product := Fp32.mul(io.a, io.b)
  io.clamped := Fp32.clampUnit(io.a)
  io.equal := Fp32.compareEq(io.a, io.b)
  io.greaterOrEqual := Fp32.compareGe(io.a, io.b)
  io.finite := Fp32.isFinite(io.a)
}

/** Tests the binary32 operations used in mechanism-path decisions. */
class Fp32Spec extends AnyFunSuite with ChiselSim {
  test("Fp32 preserves basic binary32 arithmetic and unit clamping") {
    simulate(new Fp32Harness) { dut =>
      dut.io.a.poke(fpOne.U)
      dut.io.b.poke(fpHalf.U)
      dut.clock.step()
      dut.io.sum.expect("h3fc00000".U)
      dut.io.product.expect(fpHalf.U)
      dut.io.clamped.expect(fpOne.U)

      dut.io.a.poke("hbf800000".U)
      dut.clock.step()
      dut.io.clamped.expect(fpZero.U)

      // IEEE-754 numerical equality treats the distinct signed-zero
      // encodings as equal before TRU applies its GID tie break.
      dut.io.a.poke(fpZero.U)
      dut.io.b.poke("h80000000".U)
      dut.clock.step()
      dut.io.equal.expect(true.B)

      dut.io.a.poke(fpOne.U)
      dut.io.b.poke(fpOne.U)
      dut.clock.step()
      dut.io.greaterOrEqual.expect(true.B)

      // Ordered comparisons must not turn an unordered NaN operand into a
      // successful SIU threshold decision.
      dut.io.a.poke("h7fc00000".U)
      dut.io.b.poke(fpOne.U)
      dut.clock.step()
      dut.io.greaterOrEqual.expect(false.B)
      dut.io.finite.expect(false.B)

      dut.io.a.poke("h7f800000".U)
      dut.clock.step()
      dut.io.finite.expect(false.B)

      dut.io.a.poke("h7f7fffff".U)
      dut.clock.step()
      dut.io.finite.expect(true.B)
    }
  }

  test("DivF32 returns correctly rounded binary32 quotients") {
    simulate(new DivF32) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step()

      def divide(numerator: BigInt, denominator: BigInt, expected: BigInt): Unit = {
        dut.io.in.bits.numerator.poke(numerator.U)
        dut.io.in.bits.denominator.poke(denominator.U)
        dut.io.in.valid.poke(true.B)
        stepUntil(dut.clock, 4)(dut.io.in.ready.peek().litToBoolean)
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
        stepUntil(dut.clock, 160)(dut.io.out.valid.peek().litToBoolean)
        dut.io.out.bits.expect(expected.U)
        dut.clock.step()
      }

      divide(fpOne, BigInt("40400000", 16), BigInt("3eaaaaab", 16))
      divide(BigInt("3f400000", 16), BigInt("3fc00000", 16), fpHalf)
    }
  }
}
