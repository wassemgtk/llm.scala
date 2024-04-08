package llm
//class represents the transformer model
class Transformer(config: Config) {
  private val layers = Array.fill(config.numLayers)(new TransformerLayer(config))
  private val dropout = new Dropout(config.dropoutRate)

  def forward(inputs: Array[Array[Float]]): Array[Array[Float]] = {
    var hidden = inputs
    for (layer <- layers) {
      hidden = layer.forward(hidden)
    }
    dropout.forward(hidden)
  }

  def backward(dOutputs: Array[Array[Float]]): Array[Array[Float]] = {
    var dHidden = dropout.backward(dOutputs)
    for (layer <- layers.reverse) {
      dHidden = layer.backward(dHidden)
    }
    dHidden
  }
}

class TransformerLayer(config: Config) {
  private val attention = new Attention(config)
  private val feedForward = new FeedForward(config)
  private val layerNorm1 = new LayerNorm(config.embeddingDim)
  private val layerNorm2 = new LayerNorm(config.embeddingDim)

  def forward(inputs: Array[Array[Float]]): Array[Array[Float]] = {
    val attentionOutput = attention.forward(inputs)
    val residual1 = inputs.zip(attentionOutput).map { case (input, output) =>
      vectorAdd(input, output)
    }
    val norm1 = layerNorm1.forward(residual1)
    val feedForwardOutput = feedForward.forward(norm1)
    val residual2 = norm1.zip(feedForwardOutput).map { case (input, output) =>
      vectorAdd(input, output)
    }
    layerNorm2.forward(residual2)
  }

  def backward(dOutputs: Array[Array[Float]]): Array[Array[Float]] = {
    val dResidual2 = layerNorm2.backward(dOutputs)
    val (dNorm1, dFeedForwardOutput) = dResidual2.zip(feedForward.output).map { case (dOutput, output) =>
      (dOutput, dOutput)
    }.unzip
    val dFeedForward = feedForward.backward(dFeedForwardOutput)
    val dResidual1 = layerNorm1.backward(dNorm1)
    val (dInputs, dAttentionOutput) = dResidual1.zip(attention.output).map { case (dOutput, output) =>
      (dOutput, dOutput)
    }.unzip
    val dAttention = attention.backward(dAttentionOutput)
    vectorAdd(dInputs, dAttention)
  }

  private def vectorAdd(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    v1.zip(v2).map { case (x, y) => x + y }
  }
}