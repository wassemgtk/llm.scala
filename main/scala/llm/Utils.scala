package llm

import scala.io.Source

object Utils {
  def readLinesFromFile(filePath: String): Array[String] = {
    val source = Source.fromFile(filePath)
    val lines = try source.getLines().toArray finally source.close()
    lines
  }

  def writeLinesToFile(filePath: String, lines: Array[String]): Unit = {
    val writer = new java.io.PrintWriter(new java.io.File(filePath))
    try lines.foreach(writer.println) finally writer.close()
  }

  def saveModel(model: Model, filePath: String): Unit = {
    // Implement saving the model to a file
    // You can use Java serialization or any other preferred method
  }

  def loadModel(filePath: String): Model = {
    // Implement loading the model from a file
    // You can use Java deserialization or any other preferred method
    null // Placeholder, replace with the actual loaded model
  }
}