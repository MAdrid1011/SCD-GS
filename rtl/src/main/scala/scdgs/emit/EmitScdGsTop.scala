package scdgs.emit

import java.io.File
import circt.stage.ChiselStage
import scdgs.config.ScdGsParams
import scdgs.top.ScdGsPublicTop

/** Emits public SystemVerilog only to an explicitly supplied local directory. */
object EmitScdGsTop extends App {
  private val defaultDir = new File("../out/sv")
  private val targetDir = args.sliding(2).collectFirst {
    case Array("--target-dir", value) => new File(value)
  }.getOrElse(defaultDir)
  targetDir.mkdirs()
  ChiselStage.emitSystemVerilogFile(
    new ScdGsPublicTop(ScdGsParams()),
    args = Array("--target-dir", targetDir.getPath),
    firtoolOpts = Array("--split-verilog", "--lowering-options=disallowPackedArrays,disallowLocalVariables", "--disable-all-randomization", "--strip-debug-info")
  )
}
