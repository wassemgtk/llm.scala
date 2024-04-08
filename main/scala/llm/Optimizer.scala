package llm

class AdamOptimizer(
  learningRate: Float,
  beta1: Float = 0.9f,
  beta2: Float = 0.999f,
  epsilon: Float = 1e-8f
) {
  private var m: Array[Float] = _
  private var v: Array[Float] = _
  private var t: Int = 0

  def optimize(parameters: Array[Float], gradients: Array[Float]): Unit = {
    if (m == null) {
      m = Array.fill(parameters.length)(0.0f)
      v = Array.fill(parameters.length)(0.0f)
    }

    t += 1

    for (i <- parameters.indices) {
      m(i) = beta1 * m(i) + (1 - beta1) * gradients(i)
      v(i) = beta2 * v(i) + (1 - beta2) * gradients(i) * gradients(i)

      val mHat = m(i) / (1 - math.pow(beta1, t).toFloat)
      val vHat = v(i) / (1 - math.pow(beta2, t).toFloat)

      parameters(i) -= learningRate * mHat / (math.sqrt(vHat).toFloat + epsilon)
    }
  }
}