# llm.scala
About LLM training in simple, raw Scala

### Inspired by https://github.com/karpathy/llm.c project 


```
src/
  main/
    scala/
      llm/
        Config.scala
        Model.scala
        Attention.scala
        LayerNorm.scala
        GELU.scala
        Embedding.scala
        PositionalEncoding.scala
        FeedForward.scala
        Transformer.scala
        Optimizer.scala
        DataLoader.scala
        Tokenizer.scala
        Utils.scala
        Main.scala
  test/
    scala/
      llm/
        ModelSpec.scala
        AttentionSpec.scala
        LayerNormSpec.scala
        GELUSpec.scala
        EmbeddingSpec.scala
        PositionalEncodingSpec.scala
        FeedForwardSpec.scala
        TransformerSpec.scala
        OptimizerSpec.scala
        DataLoaderSpec.scala
        TokenizerSpec.scala
        UtilsSpec.scala
data/
  tiny_shakespeare_train.bin
  tiny_shakespeare_val.bin
checkpoints/
  model_checkpoint.bin
build.sbt
README.md
```

Here's a brief description of each file:

- `Config.scala`: Contains the configuration parameters for the model.
- `Model.scala`: Defines the main model class and its forward and backward pass.
- `Attention.scala`: Implements the attention mechanism.
- `LayerNorm.scala`: Implements layer normalization.
- `GELU.scala`: Implements the GELU activation function.
- `Embedding.scala`: Implements the token and position embeddings.
- `PositionalEncoding.scala`: Implements positional encoding.
- `FeedForward.scala`: Implements the feed-forward neural network.
- `Transformer.scala`: Defines the transformer block.
- `Optimizer.scala`: Implements the optimizer for training the model.
- `DataLoader.scala`: Handles loading and batching the training data.
- `Tokenizer.scala`: Implements tokenization of input text.
- `Utils.scala`: Contains utility functions used throughout the codebase.
- `Main.scala`: The main entry point of the application.
- `ModelSpec.scala` and other `*Spec.scala` files: Contains unit tests for each module.
- `data/`: Directory to store the training and validation data.
- `checkpoints/`: Directory to store model checkpoints during training.
- `build.sbt`: The build configuration file for the Scala project.
- `README.md`: Documentation and instructions for the project.
