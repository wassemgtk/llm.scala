package llm

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TokenizerSpec extends AnyFlatSpec with Matchers {
  "Tokenizer" should "encode and decode text correctly" in {
    val vocabFile = "path/to/vocab.txt"
    val tokenizer = new Tokenizer(vocabFile)

    val text = "This is a sample text."
    val encoded = tokenizer.encode(text)
    val decoded = tokenizer.decode(encoded)

    encoded.length should be > 0
    decoded shouldBe text
  }
}