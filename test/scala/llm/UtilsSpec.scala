package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class UtilsSpec extends AnyFlatSpec with Matchers {
  "Utils" should "read lines from file correctly" in {
    val filePath = "path/to/file.txt"
    val lines = Utils.readLinesFromFile(filePath)

    lines.length should be > 0
  }

  it should "write lines to file correctly" in {
    val filePath = "path/to/output.txt"
    val lines = Array("Line 1", "Line 2", "Line 3")
    Utils.writeLinesToFile(filePath, lines)

    val readLines = Utils.readLinesFromFile(filePath)
    readLines shouldBe lines
  }

  it should "save and load model correctly" in {
    val config = Config(vocabSize = 100, embeddingDim = 32, numLayers = 2, numHeads = 4, feedForwardDim = 64)
    val model = new Model(config)

    val filePath = "path/to/model.bin"
    Utils.saveModel(model, filePath)

    val loadedModel = Utils.loadModel(filePath)
    loadedModel shouldBe a[Model]
  }
}