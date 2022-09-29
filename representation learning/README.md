




## Embedding

- **Distributed Representations of Words and Phrases and their Compositionality.** (NIPS, 2013) [[paper](http://arxiv.org/abs/1310.4546)]
- **Efficient Estimation of Word Representations in Vector Space.** (ICLR, 2013) [[paper](http://arxiv.org/abs/1301.3781)]

## Graph

- A Survey on Graph Representation Learning Methods. [[paper](https://arxiv.org/abs/2204.01855)]
- Learning Causal Effects on Hypergraphs. (KDD, 2022) [[paper](http://arxiv.org/abs/2207.04049)] (emmm; empirical; graph; hypergraph; causal)
- Feature Overcorrelation in Deep Graph Neural Networks: A New Perspective. (KDD, 2022) [[paper](http://arxiv.org/abs/2206.07743)] [[code](https://github.com/ChandlerBang/DeCorr)] (wow; empirical; graph; over-smoothing)
- GraphMAE: Self-Supervised Masked Graph Autoencoders. (KDD, 2022) [[paper](http://arxiv.org/abs/2205.10803)] [[code](https://github.com/THUDM/GraphMAE)] (novel; empirical; graph; self-supervised)
- On the Bottleneck of Graph Neural Networks and Its Practical Implications. (ICLR, 2021) [[paper](https://arxiv.org/abs/2006.05205)] [[code](https://github.com/tech-srl/bottleneck/)] (novel; empirical; graph; GNN; over-squashing)
- A Unified View on Graph Neural Networks as Graph Signal Denoising. (CIKM, 2021) [[paper](https://dl.acm.org/doi/10.1145/3459637.3482225)] [[code](https://github.com/alge24/ADA-UGNN)] (seminal; novel; theoretical; graph; GNN; denoising)
- Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs. (NIPS, 2020) [[paper](https://arxiv.org/abs/2006.11468)] [[code](https://github.com/GemsLab/H2GCN)] (novel; theoretical; graph; GNN)
- DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. (ICLR, 2020) [[paper](http://arxiv.org/abs/1907.10903)] [[code](https://github.com/DropEdge/DropEdge)] (graph; over-smoothing; dropout)
- Simplifying Graph Convolutional Networks. (ICML, 2019) [[paper](Simplifying Graph Convolutional Networks)] [[code](https://github.com/Tiiiger/SGC)] (graph; novel; GNN; empirical; SGC)
- How Powerful are Graph Neural Networks? (ICLR, 2019) [[paper](http://arxiv.org/abs/1810.00826)] [[code](https://github.com/weihua916/powerful-gnns)] (wow; seminal; theoretical; graph; WL-Test)
- Predict then Propagate: Graph Neural Networks meet Personalized PageRank. (2019) [[paper](http://arxiv.org/abs/1810.05997)] (random walk; PageRank)
- Embedding Temporal Network via Neighborhood Formation. (KDD, 2018) [[paper](https://dl.acm.org/doi/10.1145/3219819.3220054)] (empirical; novel; Hawkes Process; graph; dynamic)
- Graph Attention Networks. (ICLR, 2018) [[paper](http://arxiv.org/abs/1710.10903)] [[code](https://github.com/PetarV-/GAT)] (novel; seminal; empirical; graph; attention)
- Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. (AAAI, 2018) [[paper](http://arxiv.org/abs/1801.07606)] (讲了之前 GCN 的弊端, normalzied Laplacian 矩阵 特征值:[-1, 1])
- Dynamic Network Embedding by Modeling Triadic Closure Process. (AAAI, 2018) [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11257)] [[code](https://github.com/luckiezhou/DynamicTriad)] (emmm; graph; dynamic; empirical)
- Neural Message Passing for Quantum Chemistry. (ICML, 2017) [[paper](https://arxiv.org/abs/1704.01212)] (seminal; MPNN; GNN; novel; empirical; graph)
- Inductive Representation Learning on Large Graphs. (NIPS, 2017) [[paper](http://arxiv.org/abs/1706.02216)] [[code](https://www.cnblogs.com/MTandHJ/p/16642400.html)] (random walk; no embeddings)
- Semi-Supervised Classification with Graph Convolutional Networks. (ICLR, 2017) [[paper](http://arxiv.org/abs/1609.02907)] [[code](https://github.com/tkipf/gcn)] (GCN, seminal)
- Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. (NIPS, 2016) [[paper](https://proceedings.neurips.cc/paper/2016/hash/04df4d434d481c5bb723be1b6df1ee65-Abstract.html)] [[code](https://github.com/mdeff/cnn_graph)] (GCN, 切比雪夫核)
- Structural Deep Network Embedding. (KDD, 2016) [[paper](https://dl.acm.org/doi/10.1145/2939672.2939753)]
- node2vec: Scalable Feature Learning for Networks. (KDD, 2016) [[paper](http://arxiv.org/abs/1607.00653)] [[code](http://snap.stanford.edu/node2vec/)] [[DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch/node2vec)] (empirical; seminal; novel; graph)
- Learning with Partially Absorbing Random Walks. (NIPS, 2012) [[paper](https://proceedings.neurips.cc/paper/2012/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html)] (提出一种图聚类方法, 可以和很多现有方法产生联系, 但是这么定义的来源是什么?)

## Sparsity

- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. (ICLR, 2019) [[paper](http://arxiv.org/abs/1803.03635)] (seminal; pruning)
