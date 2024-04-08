package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FeedForwardSpec extends AnyFlatSpec with Matchers {
  "FeedForward" should "perform forward pass correctly" in {
    val config = Config(embeddingDim = 32, feedForwardDim = 64)
    val feedForward = new FeedForward(config)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = feedForward.forward(inputs)

    outputs.length shouldBe inputs.length
    outputs.head.length shouldBe config.embeddingDim
  }

  it should "perform backward pass correctly" in {
    val config = Config(embeddingDim = 32, feedForwardDim = 64)
    val feedForward = new FeedForward(config)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = feedForward.forward(inputs)
    val gradients = feedForward.backward(outputs)

    gradients.length shouldBe inputs.length
    gradients.head.length shouldBe config.embeddingDim
  }
}