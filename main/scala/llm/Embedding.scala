//the Embedding class represents the embedding layer. 
//It initializes the embedding matrix randomly using truncated normal distribution. 
//The forward method retrieves the embedding vectors for the input indices. 
//The backward method updates the embedding vectors based on the gradients.

package llm

class Embedding(config: Config) {
  private val embeddings = Array.ofDim[Float](config.vocabSize, config.embeddingDim)
  initializeEmbeddings()

  def forward(inputs: Array[Int]): Array[Array[Float]] = {
    inputs.map { input =>
      embeddings(input)
    }
  }

  def backward(dOutputs: Array[Array[Float]], inputs: Array[Int]): Unit = {
    dOutputs.zip(inputs).foreach { case (dOutput, input) =>
      embeddings(input) = vectorAdd(embeddings(input), dOutput)
    }
  }

  private def initializeEmbeddings(): Unit = {
    val stddev = 1.0 / math.sqrt(config.embeddingDim)
    embeddings.indices.foreach { i =>
      embeddings(i) = Array.fill(config.embeddingDim)(truncatedNormal(0.0, stddev))
    }
  }

  private def truncatedNormal(mean: Double, stddev: Double): Float = {
    var x = 0.0
    do {
      x = util.Random.nextGaussian() * stddev + mean
    } while (x < -2.0 * stddev || x > 2.0 * stddev)
    x.toFloat
  }

  private def vectorAdd(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    v1.zip(v2).map { case (x, y) => x + y }
  }
}