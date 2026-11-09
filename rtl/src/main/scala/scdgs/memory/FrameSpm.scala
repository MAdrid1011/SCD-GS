package scdgs.memory

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config.ScdGsParams

/**
  * Bounded target and rendered-pixel record storage alongside TUU's live
  * streams. The default 256-entry arrays are direct mapped by low pixel-key
  * bits; full tags turn a collision into a miss for the displaced record.
  * This is record retention, not a pixelwise frame-buffer implementation.
  */
class FrameSpm(p: ScdGsParams = ScdGsParams(), entries: Int = 256) extends Module {
  val io = IO(new Bundle {
    val targetWrite = Flipped(Decoupled(new TargetPixelBundle(p)))
    val targetReadPixel = Input(UInt(32.W))
    val targetRead = Output(new TargetPixelBundle(p))
    val targetValid = Output(Bool())
    val renderWrite = Flipped(Decoupled(new PixelResultBundle(p)))
    val renderReadPixel = Input(UInt(32.W))
    val renderRead = Output(new PixelResultBundle(p))
    val renderValid = Output(Bool())
  })

  private val indexWidth = math.max(1, log2Ceil(entries))
  private def index(pixel: UInt): UInt = if (entries == 1) 0.U else pixel(indexWidth - 1, 0)
  val targets = Mem(entries, new TargetPixelBundle(p))
  val renders = Mem(entries, new PixelResultBundle(p))
  val targetWritten = RegInit(VecInit(Seq.fill(entries)(false.B)))
  val renderWritten = RegInit(VecInit(Seq.fill(entries)(false.B)))
  // The compact arrays are indexed by low pixel bits, so retain full tags to
  // prevent an unrelated pixel in the same slot from reading as a hit.
  val targetTag = RegInit(VecInit(Seq.fill(entries)(0.U(32.W))))
  val renderTag = RegInit(VecInit(Seq.fill(entries)(0.U(32.W))))

  io.targetWrite.ready := true.B
  when(io.targetWrite.fire) {
    targets.write(index(io.targetWrite.bits.pixel), io.targetWrite.bits)
    targetWritten(index(io.targetWrite.bits.pixel)) := true.B
    targetTag(index(io.targetWrite.bits.pixel)) := io.targetWrite.bits.pixel
  }
  io.renderWrite.ready := true.B
  when(io.renderWrite.fire) {
    renders.write(index(io.renderWrite.bits.pixel), io.renderWrite.bits)
    renderWritten(index(io.renderWrite.bits.pixel)) := true.B
    renderTag(index(io.renderWrite.bits.pixel)) := io.renderWrite.bits.pixel
  }
  io.targetRead := targets.read(index(io.targetReadPixel))
  io.targetValid := targetWritten(index(io.targetReadPixel)) && targetTag(index(io.targetReadPixel)) === io.targetReadPixel
  io.renderRead := renders.read(index(io.renderReadPixel))
  io.renderValid := renderWritten(index(io.renderReadPixel)) && renderTag(index(io.renderReadPixel)) === io.renderReadPixel
}
