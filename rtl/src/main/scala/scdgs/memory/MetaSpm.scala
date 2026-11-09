package scdgs.memory

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config.ScdGsParams

/** Published/readback snapshot of FRU metadata, not the live execution owner. */
class MetaSpm(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val write = Flipped(Decoupled(new MetaEntryBundle(p)))
    val writeGid = Input(UInt(p.gidWidth.W))
    val readGid = Input(UInt(p.gidWidth.W))
    val read = Output(new MetaEntryBundle(p))
    val readValid = Output(Bool())
  })

  private val indexWidth = math.max(1, log2Ceil(p.maxGaussians))
  private def index(gid: UInt): UInt = if (p.maxGaussians == 1) 0.U else gid(indexWidth - 1, 0)
  private def valid(gid: UInt): Bool = gid < p.maxGaussians.U
  val entries = Mem(p.maxGaussians, new MetaEntryBundle(p))
  val written = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  io.write.ready := true.B
  when(io.write.fire && valid(io.writeGid)) {
    entries.write(index(io.writeGid), io.write.bits)
    written(index(io.writeGid)) := true.B
  }
  io.read := Mux(valid(io.readGid), entries.read(index(io.readGid)), 0.U.asTypeOf(new MetaEntryBundle(p)))
  io.readValid := valid(io.readGid) && written(index(io.readGid))
}
