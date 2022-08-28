




## Embedding

- **Distributed Representations of Words and Phrases and their Compositionality.** (NIPS, 2013) [[paper](http://arxiv.org/abs/1310.4546)]
- **Efficient Estimation of Word Representations in Vector Space.** (ICLR, 2013) [[paper](http://arxiv.org/abs/1301.3781)]

## Graph

- DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. (ICLR, 2020) [[paper](http://arxiv.org/abs/1907.10903)] [[code](https://github.com/DropEdge/DropEdge)] (graph; over-smoothing; dropout)
- Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. (AAAI, 2018) [[paper](http://arxiv.org/abs/1801.07606)] (讲了之前 GCN 的弊端, normalzied Laplacian 矩阵 特征值:[-1, 1])
- Semi-Supervised Classification with Graph Convolutional Networks. (ICLR, 2017) [[paper](http://arxiv.org/abs/1609.02907)] [[code](https://github.com/tkipf/gcn)] (GCN, seminal)
- Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. (NIPS, 2016) [[paper](https://proceedings.neurips.cc/paper/2016/hash/04df4d434d481c5bb723be1b6df1ee65-Abstract.html)] [[code](https://github.com/mdeff/cnn_graph)] (GCN, 切比雪夫核)
- Learning with Partially Absorbing Random Walks. (NIPS, 2012) [[paper](https://proceedings.neurips.cc/paper/2012/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html)] (提出一种图聚类方法, 可以和很多现有方法产生联系, 但是这么定义的来源是什么?)

## Sparsity

- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. (ICLR, 2019) [[paper](http://arxiv.org/abs/1803.03635)] (seminal; pruning)
