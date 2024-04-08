package llm

class Attention(config: Config) {
  private val queryMatrix = new Linear(config.embeddingDim, config.embeddingDim)
  private val keyMatrix = new Linear(config.embeddingDim, config.embeddingDim)
  private val valueMatrix = new Linear(config.embeddingDim, config.embeddingDim)
  private val outputMatrix = new Linear(config.embeddingDim, config.embeddingDim)
  private val dropout = new Dropout(config.dropoutRate)

  def forward(inputs: Array[Array[Float]]): Array[Array[Float]] = {
    val batchSize = inputs.length
    val seqLen = inputs.head.length

    val queries = queryMatrix.forward(inputs)
    val keys = keyMatrix.forward(inputs)
    val values = valueMatrix.forward(inputs)

    val scores = Array.ofDim[Float](batchSize, config.numHeads, seqLen, seqLen)
    for (b <- 0 until batchSize; h <- 0 until config.numHeads; i <- 0 until seqLen; j <- 0 until seqLen) {
      scores(b)(h)(i)(j) = dotProduct(queries(b)(i), keys(b)(j)) / math.sqrt(config.embeddingDim)
    }

    val weights = scores.map(softmax)
    val droppedWeights = dropout.forward(weights)

    val weightedValues = Array.ofDim[Float](batchSize, config.numHeads, seqLen, config.embeddingDim)
    for (b <- 0 until batchSize; h <- 0 until config.numHeads; i <- 0 until seqLen; j <- 0 until seqLen) {
      weightedValues(b)(h)(i) = vectorAdd(weightedValues(b)(h)(i), vectorScale(values(b)(j), droppedWeights(b)(h)(i)(j)))
    }

    val concatValues = weightedValues.map(_.transpose.flatten)
    outputMatrix.forward(concatValues)
  }

  def backward(dOutputs: Array[Array[Float]]): Array[Array[Float]] = {
    val batchSize = dOutputs.length
    val seqLen = dOutputs.head.length / config.embeddingDim

    val dConcatValues = outputMatrix.backward(dOutputs)
    val dWeightedValues = dConcatValues.map(_.grouped(config.embeddingDim).toArray.transpose)

    val dValues = Array.ofDim[Float](batchSize, seqLen, config.embeddingDim)
    val dDroppedWeights = Array.ofDim[Float](batchSize, config.numHeads, seqLen, seqLen)
    for (b <- 0 until batchSize; h <- 0 until config.numHeads; i <- 0 until seqLen; j <- 0 until seqLen) {
      dValues(b)(j) = vectorAdd(dValues(b)(j), vectorScale(dWeightedValues(b)(h)(i), dropout.mask(b)(h)(i)(j)))
      dDroppedWeights(b)(h)(i)(j) = dotProduct(dWeightedValues(b)(h)(i), valueMatrix.output(b)(j))
    }

    val dWeights = dropout.backward(dDroppedWeights)
    val dScores = dWeights.map(softmaxBackward)

    val dQueries = Array.ofDim[Float](batchSize, seqLen, config.embeddingDim)
    val dKeys = Array.ofDim[Float](batchSize, seqLen, config.embeddingDim)
    for (b <- 0 until batchSize; h <- 0 until config.numHeads; i <- 0 until seqLen; j <- 0 until seqLen) {
      val dScore = dScores(b)(h)(i)(j) / math.sqrt(config.embeddingDim)
      dQueries(b)(i) = vectorAdd(dQueries(b)(i), vectorScale(keyMatrix.output(b)(j), dScore))
      dKeys(b)(j) = vectorAdd(dKeys(b)(j), vectorScale(queryMatrix.output(b)(i), dScore))
    }

    val dInputs = queryMatrix.backward(dQueries)
    keyMatrix.backward(dKeys)
    valueMatrix.backward(dValues)

    dInputs
  }

  private def dotProduct(v1: Array[Float], v2: Array[Float]): Float = {
    v1.zip(v2).map { case (x, y) => x * y }.sum
  }

  private def vectorAdd(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    v1.zip(v2).map { case (x, y) => x + y }
  }

  private def vectorScale(v: Array[Float], s: Float): Array[Float] = {
    v.map(_ * s)
  }

  private def softmax(scores: Array[Array[Float]]): Array[Array[Float]] = {
    val maxScores = scores.map(_.max)
    val expScores = scores.zip(maxScores).map { case (row, maxScore) =>
      row.map(score => math.exp(score - maxScore).toFloat)
    }
    val sumExpScores = expScores.map(_.sum)
    expScores.zip(sumExpScores).map { case (row, sum) =>
      row.map(_ / sum)
    }
  }

  private def softmaxBackward(dOutputs: Array[Array[Float]]): Array[Array[Float]] = {
    val softmaxOutputs = dOutputs.map(softmax)
    dOutputs.zip(softmaxOutputs).map { case (dOutput, output) =>
      output.zip(dOutput).map { case (o, d) =>
        o * (d - dotProduct(output, dOutput))
      }
    }
  }
}