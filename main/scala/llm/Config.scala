package llm

case class Config(
  maxSeqLen: Int = 1024,
  vocabSize: Int = 50257,
  numLayers: Int = 12,
  numHeads: Int = 12,
  embeddingDim: Int = 768,
  feedForwardDim: Int = 3072,
  initStd: Double = 0.02,
  dropoutRate: Double = 0.1
)