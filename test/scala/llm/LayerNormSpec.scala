package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LayerNormSpec extends AnyFlatSpec with Matchers {
  "LayerNorm" should "perform forward pass correctly" in {
    val layerNorm = new LayerNorm(32)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = layerNorm.forward(inputs)

    outputs.length shouldBe inputs.length
    outputs.head.length shouldBe inputs.head.length
  }

  it should "perform backward pass correctly" in {
    val layerNorm = new LayerNorm(32)

    val inputs = Array(
      Array.fill(32)(1.0f),
      Array.fill(32)(2.0f),
      Array.fill(32)(3.0f)
    )

    val outputs = layerNorm.forward(inputs)
    val gradients = layerNorm.backward(outputs)

    gradients.length shouldBe inputs.length
    gradients.head.length shouldBe inputs.head.length
  }
}