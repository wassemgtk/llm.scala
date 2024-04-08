//the forward method implements the Gaussian Error Linear Unit (GELU) activation function. 
//It applies the GELU function element-wise to the input array. 
//The backward method computes the gradients of the GELU function with respect to the input.
package llm

object GELU {
  def forward(x: Array[Float]): Array[Float] = {
    val sqrt2OverPi = 0.7978845608028654 // sqrt(2 / pi)
    x.map { xi =>
      0.5f * xi * (1.0f + tanh(sqrt2OverPi * (xi + 0.044715f * pow(xi, 3))))
    }
  }

  def backward(dOutputs: Array[Float], inputs: Array[Float]): Array[Float] = {
    val sqrt2OverPi = 0.7978845608028654 // sqrt(2 / pi)
    dOutputs.zip(inputs).map { case (dOutput, input) =>
      val cdf = 0.5f * (1.0f + tanh(sqrt2OverPi * (input + 0.044715f * pow(input, 3))))
      val pdf = 0.398942280401432677939946059934 * exp(-0.5f * pow(input, 2)) // 1 / sqrt(2 * pi) * exp(-x^2 / 2)
      dOutput * (cdf + input * pdf)
    }
  }

  private def tanh(x: Float): Float = {
    val expX = exp(x)
    val expNegX = exp(-x)
    (expX - expNegX) / (expX + expNegX)
  }

  private def exp(x: Float): Float = {
    math.exp(x).toFloat
  }

  private def pow(x: Float, p: Float): Float = {
    math.pow(x, p).toFloat
  }
}