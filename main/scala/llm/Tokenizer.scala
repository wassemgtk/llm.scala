package llm

import scala.collection.mutable

class Tokenizer(
  val vocabFile: String,
  val unknownToken: String = "<unk>",
  val padToken: String = "<pad>",
  val startToken: String = "<start>",
  val endToken: String = "<end>"
) {
  private val tokenToId: mutable.Map[String, Int] = mutable.Map[String, Int]()
  private val idToToken: mutable.Map[Int, String] = mutable.Map[Int, String]()

  buildVocab()

  private def buildVocab(): Unit = {
    val vocab = Utils.readLinesFromFile(vocabFile)
    vocab.zipWithIndex.foreach { case (token, id) =>
      tokenToId(token) = id
      idToToken(id) = token
    }
    tokenToId(unknownToken) = vocab.length
    tokenToId(padToken) = vocab.length + 1
    tokenToId(startToken) = vocab.length + 2
    tokenToId(endToken) = vocab.length + 3
  }

  def encode(text: String): Array[Int] = {
    val tokens = text.split("\\s+")
    tokens.map(token => tokenToId.getOrElse(token, tokenToId(unknownToken)))
  }

  def decode(ids: Array[Int]): String = {
    ids.map(id => idToToken.getOrElse(id, unknownToken)).mkString(" ")
  }
}