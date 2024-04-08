//class generates positional encodings for the input sequence

package llm

class PositionalEncoding(config: Config) {
  private val encodings = Array.ofDim[Float](config.maxSeqLen, config.embeddingDim)
  initializeEncodings()

  def forward(seqLen: Int): Array[Array[Float]] = {
    encodings.take(seqLen)
  }

  private def initializeEncodings(): Unit = {
    for (pos <- 0 until config.maxSeqLen; i <- 0 until config.embeddingDim) {
      val angle = pos / math.pow(10000, 2 * (i / 2) / config.embeddingDim)
      encodings(pos)(i) = if (i % 2 == 0) sin(angle) else cos(angle)
    }
  }

  private def sin(x: Double): Float = {
    math.sin(x).toFloat
  }

  private def cos(x: Double): Float = {
    math.cos(x).toFloat
  }
}