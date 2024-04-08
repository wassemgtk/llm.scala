package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DataLoaderSpec extends AnyFlatSpec with Matchers {
  "DataLoader" should "load data and provide batches" in {
    val filePath = "path/to/data.txt"
    val batchSize = 2
    val seqLen = 4
    val tokenizer = new Tokenizer("path/to/vocab.txt")
    val dataLoader = new DataLoader(filePath, batchSize, seqLen, tokenizer)

    val (inputs, targets) = dataLoader.nextBatch()

    inputs.length shouldBe batchSize
    inputs.head.length shouldBe seqLen
    targets.length shouldBe batchSize
    targets.head.length shouldBe seqLen
  }
}