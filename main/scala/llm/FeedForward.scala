//Class represents the feed-forward neural network layer

package llm

class FeedForward(config: Config) {
  private val linear1 = new Linear(config.embeddingDim, config.feedForwardDim)
  private val linear2 = new Linear(config.feedForwardDim, config.embeddingDim)
  private val dropout = new Dropout(config.dropoutRate)

  def forward(inputs: Array[Array[Float]]): Array[Array[Float]] = {
    val hidden = linear1.forward(inputs)
    val activated = hidden.map(GELU.forward)
    val dropped = dropout.forward(activated)
    linear2.forward(dropped)
  }

  def backward(dOutputs: Array[Array[Float]]): Array[Array[Float]] = {
    val dDropped = linear2.backward(dOutputs)
    val dActivated = dropout.backward(dDropped)
    val dHidden = dActivated.zip(linear1.output).map { case (dOutput, input) =>
      GELU.backward(dOutput, input)
    }
    linear1.backward(dHidden)
  }
}