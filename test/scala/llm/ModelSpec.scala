package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ModelSpec extends AnyFlatSpec with Matchers {
  "Model" should "perform forward pass correctly" in {
    val config = Config(vocabSize = 100, embeddingDim = 32, numLayers = 2, numHeads = 4, feedForwardDim = 64)
    val model = new Model(config)

    val inputs = Array(
      Array(1, 2, 3, 4),
      Array(5, 6, 7, 8)
    )
    val targets = Array(
      Array(2, 3, 4, 5),
      Array(6, 7, 8, 9)
    )

    val (logits, loss) = model.forward(inputs, Some(targets))

    logits.length shouldBe 2
    logits.head.length shouldBe 4
    logits.head.head.length shouldBe config.vocabSize

    loss shouldBe a[Float]
  }

  it should "perform backward pass correctly" in {
    val config = Config(vocabSize = 100, embeddingDim = 32, numLayers = 2, numHeads = 4, feedForwardDim = 64)
    val model = new Model(config)

    val inputs = Array(
      Array(1, 2, 3, 4),
      Array(5, 6, 7, 8)
    )
    val targets = Array(
      Array(2, 3, 4, 5),
      Array(6, 7, 8, 9)
    )

    val (logits, _) = model.forward(inputs, Some(targets))
    val gradients = model.backward(logits)

    gradients.length shouldBe model.parameters.length
  }
}