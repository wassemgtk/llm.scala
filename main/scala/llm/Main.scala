package llm

object Main {
  def main(args: Array[String]): Unit = {
    val configFile = "path/to/config.json"
    val config = Config.fromFile(configFile)

    val vocabFile = "path/to/vocab.txt"
    val tokenizer = new Tokenizer(vocabFile)

    val dataFile = "path/to/data.txt"
    val dataLoader = new DataLoader(dataFile, config.batchSize, config.seqLen, tokenizer)

    val model = new Model(config)
    val optimizer = new AdamOptimizer(config.learningRate)

    val numEpochs = 10
    val numSteps = 1000

    for (epoch <- 1 to numEpochs) {
      println(s"Epoch $epoch")

      for (step <- 1 to numSteps) {
        val (inputs, targets) = dataLoader.nextBatch()
        val (logits, loss) = model.forward(inputs, Some(targets))
        val gradients = model.backward(logits)
        optimizer.optimize(model.parameters, gradients)

        if (step % 100 == 0) {
          println(s"Step $step, Loss: $loss")
        }
      }

      val prompt = "Sample text"
      val generatedText = model.generate(tokenizer.encode(prompt), maxNewTokens = 100)
      println(s"Generated text: ${tokenizer.decode(generatedText)}")

      Utils.saveModel(model, s"path/to/model_epoch$epoch.bin")
    }
  }
}