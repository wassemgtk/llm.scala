package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OptimizerSpec extends AnyFlatSpec with Matchers {
  "AdamOptimizer" should "update parameters correctly" in {
    val learningRate = 0.001f
    val optimizer = new AdamOptimizer(learningRate)

    val parameters = Array(1.0f, 2.0f, 3.0f)
    val gradients = Array(0.1f, 0.2f, 0.3f)

    optimizer.optimize(parameters, gradients)

    parameters.length shouldBe 3
    parameters.head should not be 1.0f
  }
}