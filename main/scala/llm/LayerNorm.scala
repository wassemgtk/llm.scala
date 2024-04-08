package llm

class LayerNorm(dim: Int, eps: Double = 1e-5) {
  private val gamma = Array.fill(dim)(1.0f)
  private val beta = Array.fill(dim)(0.0f)

  def forward(inputs: Array[Array[Float]]): Array[Array[Float]] = {
    val mean = inputs.map(_.sum / dim)
    val variance = inputs.zip(mean).map { case (input, m) =>
      input.map(x => math.pow(x - m, 2).toFloat).sum / dim
    }
    val stddev = variance.map(v => math.sqrt(v + eps).toFloat)

    inputs.zip(mean).zip(stddev).map { case ((input, m), s) =>
      input.zip(gamma).zip(beta).map { case ((x, g), b) => (x - m) / s * g + b }
    }
  }

  def backward(dOutputs: Array[Array[Float]]): Array[Array[Float]] = {
    val batchSize = dOutputs.length
    val mean = dOutputs.map(_.sum / dim)
    val variance = dOutputs.zip(mean).map { case (dOutput, m) =>
      dOutput.map(x => math.pow(x - m, 2).toFloat).sum / dim
    }
    val stddev = variance.map(v => math.sqrt(v + eps).toFloat)

    val dGamma = dOutputs.zip(mean).zip(stddev).map { case ((dOutput, m), s) =>
      dOutput.zip(gamma).map { case (d, g) => d * g }.sum
    }
    val dBeta = dOutputs.map(_.sum)

    val dInputs = dOutputs.zip(mean).zip(stddev).map { case ((dOutput, m), s) =>
      dOutput.zip(gamma).map { case (d, g) =>
        g * (d - dBeta.sum / batchSize - (d - m) * dGamma.sum / (batchSize * s))
      }
    }

    dInputs
  }
}