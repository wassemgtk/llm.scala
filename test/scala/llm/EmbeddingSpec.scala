package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmbeddingSpec extends AnyFlatSpec with Matchers {
  "Embedding" should "perform forward pass correctly" in {
    val config = Config(vocabSize = 100, embeddingDim = 32)
    val embedding = new Embedding(config)

    val inputs = Array(1, 2, 3, 4)

    val outputs = embedding.forward(inputs)

    outputs.length shouldBe inputs.length
    outputs.head.length shouldBe config.embeddingDim
  }

  it should "perform backward pass correctly" in {
    val config = Config(vocabSize = 100, embeddingDim = 32)
    val embedding = new Embedding(config)

    val inputs = Array(1, 2, 3, 4)

    val outputs = embedding.forward(inputs)
    val gradients = Array.fill(outputs.length, outputs.head.length)(1.0f)
    embedding.backward(gradients, inputs)
  }
}