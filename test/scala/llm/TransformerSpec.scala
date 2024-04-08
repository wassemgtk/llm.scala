package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TransformerSpec extends AnyFlatSpec with Matchers {
  "Transformer" should "perform forward pass correctly" in {
    val config = Config(embeddingDim = 32, numHeads = 4, feedForwardDim = 64, numLayers = 2)
    val transformer = new Transformer(config)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = transformer.forward(inputs)

    outputs.length shouldBe inputs.length
    outputs.head.length shouldBe config.embeddingDim
  }

  it should "perform backward pass correctly" in {
    val config = Config(embeddingDim = 32, numHeads = 4, feedForwardDim = 64, numLayers = 2)
    val transformer = new Transformer(config)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = transformer.forward(inputs)
    val gradients = transformer.backward(outputs)

    gradients.length shouldBe inputs.length
    gradients.head.length shouldBe config.embeddingDim
  }
}