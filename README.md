# LLM Training in Scala

This project is an implementation of a Language Model (LLM) training framework in Scala. It provides a set of modules and utilities for building, training, and evaluating language models using the transformer architecture.

Inspired by the [llm.c](https://github.com/karpathy/llm.c) project, this Scala version aims to provide a clean, efficient, and extensible codebase for training language models.

## Features

- Transformer-based language model architecture
- Multi-head self-attention mechanism
- Positional encoding for sequence information
- Feed-forward neural network layers
- Embedding layer for input tokens
- Layer normalization for stable training
- GELU activation function
- Adam optimizer for parameter updates
- Data loading and batching utilities
- Tokenization and vocabulary handling
- Test suite for all modules

## Project Structure

The project follows a standard Scala project structure:

```
llm-training/
  ├── build.sbt
  └── src/
      ├── main/
      │   └── scala/
      │       └── llm/
      │           ├── Config.scala
      │           ├── Model.scala
      │           ├── Attention.scala
      │           ├── LayerNorm.scala
      │           ├── GELU.scala
      │           ├── Embedding.scala
      │           ├── PositionalEncoding.scala
      │           ├── FeedForward.scala
      │           ├── Transformer.scala
      │           ├── Optimizer.scala
      │           ├── DataLoader.scala
      │           ├── Tokenizer.scala
      │           ├── Utils.scala
      │           └── Main.scala
      └── test/
          └── scala/
              └── llm/
                  ├── ModelSpec.scala
                  ├── AttentionSpec.scala
                  ├── LayerNormSpec.scala
                  ├── GELUSpec.scala
                  ├── EmbeddingSpec.scala
                  ├── PositionalEncodingSpec.scala
                  ├── FeedForwardSpec.scala
                  ├── TransformerSpec.scala
                  ├── OptimizerSpec.scala
                  ├── DataLoaderSpec.scala
                  ├── TokenizerSpec.scala
                  └── UtilsSpec.scala
```

- `src/main/scala/llm/`: Contains the main source code for the language model implementation.
- `src/test/scala/llm/`: Contains the test specifications for each module.
- `build.sbt`: The build configuration file for the Scala project.
- `project/`: Contains the sbt version and plugin configuration.

## Getting Started

### Prerequisites

- Scala 2.13.8
- sbt 1.5.5

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/wassemgtk/llm.scala.git
   ```

2. Navigate to the project directory:
   ```
   cd llm-training
   ```

3. Compile the project:
   ```
   sbt compile
   ```

### Training

To train the language model, follow these steps:

1. Prepare your training data:
   - Place your training data file (e.g., `tiny_shakespeare_train.bin`) in the `data/` directory.
   - Update the `dataFile` value in `Main.scala` to point to your training data file.

2. Configure the model hyperparameters in the `Config` case class in `Config.scala`.

3. Run the training script:
   ```
   sbt run
   ```

4. Monitor the training progress and metrics logged to the console.

### Text Generation

To generate text using a trained model, follow these steps:

1. Make sure you have a trained model checkpoint in the `checkpoints/` directory.

2. Update the `modelCheckpoint` value in `Main.scala` to point to your trained model checkpoint file.

3. Set the desired generation parameters (e.g., `maxNewTokens`, `temperature`) in the `Main` object.

4. Run the text generation script:
   ```
   sbt run
   ```

5. The generated text will be printed to the console.

### Testing

To run the test suite and ensure the correctness of the implemented modules, use the following command:

```
sbt test
```

This will execute all the test specifications in the `src/test/scala/llm/` directory.

## Configuration

The `Config` case class in `src/main/scala/llm/Config.scala` contains the hyperparameters and configuration settings for the language model. You can modify these values to experiment with different model architectures and training setups.

## Model Checkpointing

During training, the model checkpoints will be saved in the `checkpoints/` directory. You can use these checkpoints to resume training from a previous state or to generate text using a trained model.

## Logging

The project uses the Logback logging library for logging purposes. You can configure the logging settings in the `src/main/resources/logback.xml` file.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- This project is inspired by the [llm.c](https://github.com/karpathy/llm.c) project by Andrej Karpathy.
- The transformer architecture is based on the paper "Attention Is All You Need" by Vaswani et al.
- The implementation draws inspiration from various open-source language model implementations in the Scala ecosystem.

