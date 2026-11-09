package scdgs.memory

import chisel3._
import chisel3.util._
import scdgs.bundle._
import scdgs.config.ScdGsParams

/** Retains prior training projections and forward trajectories by Gaussian ID. */
class SplatSpm(p: ScdGsParams = ScdGsParams()) extends Module {
  val io = IO(new Bundle {
    val summaryWrite = Flipped(Decoupled(new ProjectionSummaryBundle(p)))
    val summaryReadGid = Input(UInt(p.gidWidth.W))
    val summaryRead = Output(new ProjectionSummaryBundle(p))
    val summaryValid = Output(Bool())
    val traceWrite = Flipped(Decoupled(new RenderTraceBundle(p)))
    val traceReadGid = Input(UInt(p.gidWidth.W))
    val traceRead = Output(new RenderTraceBundle(p))
    val traceValid = Output(Bool())
  })

  private val indexWidth = math.max(1, log2Ceil(p.maxGaussians))
  private def index(gid: UInt): UInt = if (p.maxGaussians == 1) 0.U else gid(indexWidth - 1, 0)
  private def valid(gid: UInt): Bool = gid < p.maxGaussians.U

  val summaries = Mem(p.maxGaussians, new ProjectionSummaryBundle(p))
  val traces = Mem(p.maxGaussians, new RenderTraceBundle(p))
  val summaryWritten = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))
  val traceWritten = RegInit(VecInit(Seq.fill(p.maxGaussians)(false.B)))

  io.summaryWrite.ready := true.B
  when(io.summaryWrite.fire && valid(io.summaryWrite.bits.gid)) {
    summaries.write(index(io.summaryWrite.bits.gid), io.summaryWrite.bits)
    summaryWritten(index(io.summaryWrite.bits.gid)) := true.B
  }
  io.traceWrite.ready := true.B
  when(io.traceWrite.fire && valid(io.traceWrite.bits.gid)) {
    traces.write(index(io.traceWrite.bits.gid), io.traceWrite.bits)
    traceWritten(index(io.traceWrite.bits.gid)) := true.B
  }

  io.summaryRead := Mux(valid(io.summaryReadGid), summaries.read(index(io.summaryReadGid)), 0.U.asTypeOf(new ProjectionSummaryBundle(p)))
  io.summaryValid := valid(io.summaryReadGid) && summaryWritten(index(io.summaryReadGid))
  io.traceRead := Mux(valid(io.traceReadGid), traces.read(index(io.traceReadGid)), 0.U.asTypeOf(new RenderTraceBundle(p)))
  io.traceValid := valid(io.traceReadGid) && traceWritten(index(io.traceReadGid))
}
