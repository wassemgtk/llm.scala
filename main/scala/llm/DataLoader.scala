package llm

import scala.io.Source

class DataLoader(
  val filePath: String,
  val batchSize: Int,
  val seqLen: Int,
  val tokenizer: Tokenizer
) {
  private val data: Array[Int] = loadData()
  private var cursor: Int = 0

  private def loadData(): Array[Int] = {
    val source = Source.fromFile(filePath)
    val text = try source.mkString finally source.close()
    tokenizer.encode(text)
  }

  def nextBatch(): (Array[Array[Int]], Array[Array[Int]]) = {
    val batchInputs = Array.ofDim[Int](batchSize, seqLen)
    val batchTargets = Array.ofDim[Int](batchSize, seqLen)

    for (i <- 0 until batchSize) {
      if (cursor + seqLen >= data.length) {
        cursor = 0
      }
      val inputs = data.slice(cursor, cursor + seqLen)
      val targets = data.slice(cursor + 1, cursor + seqLen + 1)
      batchInputs(i) = inputs
      batchTargets(i) = targets
      cursor += seqLen
    }

    (batchInputs, batchTargets)
  }
}