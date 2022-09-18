




## Embedding

- **Distributed Representations of Words and Phrases and their Compositionality.** (NIPS, 2013) [[paper](http://arxiv.org/abs/1310.4546)]
- **Efficient Estimation of Word Representations in Vector Space.** (ICLR, 2013) [[paper](http://arxiv.org/abs/1301.3781)]

## Graph

- Learning Causal Effects on Hypergraphs. (KDD, 2022) [[paper](http://arxiv.org/abs/2207.04049)] (emmm; empirical; graph; hypergraph; causal)
- Feature Overcorrelation in Deep Graph Neural Networks: A New Perspective. (KDD, 2022) [[paper](http://arxiv.org/abs/2206.07743)] [[code](https://github.com/ChandlerBang/DeCorr)] (wow; empirical; graph; over-smoothing)
- GraphMAE: Self-Supervised Masked Graph Autoencoders. (KDD, 2022) [[paper](http://arxiv.org/abs/2205.10803)] [[code](https://github.com/THUDM/GraphMAE)] (novel; empirical; graph; self-supervised)
- DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. (ICLR, 2020) [[paper](http://arxiv.org/abs/1907.10903)] [[code](https://github.com/DropEdge/DropEdge)] (graph; over-smoothing; dropout)
- How Powerful are Graph Neural Networks? (ICLR, 2019) [[paper](http://arxiv.org/abs/1810.00826)] [[code](https://github.com/weihua916/powerful-gnns)] (wow; seminal; theoretical; graph; WL-Test)
- Predict then Propagate: Graph Neural Networks meet Personalized PageRank. (2019) [[paper](http://arxiv.org/abs/1810.05997)] (random walk; PageRank)
- Graph Attention Networks. (ICLR, 2018) [[paper](http://arxiv.org/abs/1710.10903)] [[code](https://github.com/PetarV-/GAT)] (novel; seminal; empirical; graph; attention)
- Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. (AAAI, 2018) [[paper](http://arxiv.org/abs/1801.07606)] (讲了之前 GCN 的弊端, normalzied Laplacian 矩阵 特征值:[-1, 1])
- Inductive Representation Learning on Large Graphs. (NIPS, 2017) [[paper](http://arxiv.org/abs/1706.02216)] [[code](https://www.cnblogs.com/MTandHJ/p/16642400.html)] (random walk; no embeddings)
- Semi-Supervised Classification with Graph Convolutional Networks. (ICLR, 2017) [[paper](http://arxiv.org/abs/1609.02907)] [[code](https://github.com/tkipf/gcn)] (GCN, seminal)
- Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. (NIPS, 2016) [[paper](https://proceedings.neurips.cc/paper/2016/hash/04df4d434d481c5bb723be1b6df1ee65-Abstract.html)] [[code](https://github.com/mdeff/cnn_graph)] (GCN, 切比雪夫核)
- Learning with Partially Absorbing Random Walks. (NIPS, 2012) [[paper](https://proceedings.neurips.cc/paper/2012/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html)] (提出一种图聚类方法, 可以和很多现有方法产生联系, 但是这么定义的来源是什么?)

## Sparsity

- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. (ICLR, 2019) [[paper](http://arxiv.org/abs/1803.03635)] (seminal; pruning)