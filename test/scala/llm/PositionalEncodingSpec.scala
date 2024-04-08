package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PositionalEncodingSpec extends AnyFlatSpec with Matchers {
  "PositionalEncoding" should "generate correct encodings" in {
    val config = Config(maxSeqLen = 100, embeddingDim = 32)
    val positionalEncoding = new PositionalEncoding(config)

    val seqLen = 10
    val encodings = positionalEncoding.forward(seqLen)

    encodings.length shouldBe seqLen
    encodings.head.length shouldBe config.embeddingDim
  }
}