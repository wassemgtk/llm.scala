package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GELUSpec extends AnyFlatSpec with Matchers {
  "GELU" should "perform forward pass correctly" in {
    val inputs = Array(1.0f, 2.0f, 3.0f)

    val outputs = GELU.forward(inputs)

    outputs.length shouldBe inputs.length
  }

  it should "perform backward pass correctly" in {
    val inputs = Array(1.0f, 2.0f, 3.0f)

    val outputs = GELU.forward(inputs)
    val gradients = GELU.backward(outputs, inputs)

    gradients.length shouldBe inputs.length
  }
}