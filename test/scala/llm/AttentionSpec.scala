package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AttentionSpec extends AnyFlatSpec with Matchers {
  "Attention" should "perform forward pass correctly" in {
    val config = Config(embeddingDim = 32, numHeads = 4)
    val attention = new Attention(config)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = attention.forward(inputs)

    outputs.length shouldBe inputs.length
    outputs.head.length shouldBe config.embeddingDim
  }

  it should "perform backward pass correctly" in {
    val config = Config(embeddingDim = 32, numHeads = 4)
    val attention = new Attention(config)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = attention.forward(inputs)
    val gradients = attention.backward(outputs)

    gradients.length shouldBe inputs.length
    gradients.head.length shouldBe config.embeddingDim
  }
}