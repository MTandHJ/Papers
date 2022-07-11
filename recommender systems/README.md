





## Fairness

- CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems. (SIGIR, 2022) [[paper](https://arxiv.org/abs/2204.08085)]
- Fairness among New Items in Cold Start Recommender Systems. (SIGIR, 2021) [[paper](https://dl.acm.org/doi/10.1145/3404835.3462948)]
- Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System. (SIGKDD, 2021) [[paper](http://arxiv.org/abs/2010.15363)]
- FairRec: Fairness-aware News Recommendation with Decomposed Adversarial Learning. (AAAI, 2021) [[paper](https://arxiv.org/abs/2006.16742)]
- A Framework for Understanding Sources of Harm throughout the Machine Learning Life. [[paper](http://arxiv.org/abs/1901.10002)]
- Beyond Parity: Fairness Objectives for Collaborative Filtering. (NIPS, 2017) [[paper](http://arxiv.org/abs/1705.08804)]  提出了几种 Fairness 指标, 可以直接作为一般训练方法的正则项 (文中基于最普通的协同过滤).
- Counterfactual Fairness. (NIPS, 2017) [[paper](https://papers.nips.cc/paper/2017/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html)] 定义了一种 Counterfactual Fairness, 但是除此之外没看出亮点.
- Ranking with Fairness Constraints. [[paper](http://arxiv.org/abs/1704.06840)] 本文讨论在一种'强硬'的 Fairness 约束下, 如何 (快速) re-ranking 以保留尽可能多的收益.



## Diversity

- Neural Re-ranking in Multi-stage Recommender Systems: A Review. (arXiv preprint arXiv:2202.06602) [[paper](http://arxiv.org/abs/2202.06602)] [[code](https://github.com/LibRerank-Community/LibRerank)]
- Rabbit Holes and Taste Distortion: Distribution-Aware Recommendation with Evolving Interests. (WWW, 2021) [[paper](https://people.engr.tamu.edu/caverlee/pubs/WWW21-Final-Publication.pdf)]

## Privacy

- Graph Embedding for Recommendation against Attribute Inference Attacks. (WWW, 2021) [[paper](http://arxiv.org/abs/2101.12549)]


## Adversarial Robustness

- Distributionally-robust Recommendations for Improving Worst-case User Experience. (WWW, 2022) [[paper](https://dl.acm.org/doi/10.1145/3485447.3512255)]
- Learning to Augment for Casual User Recommendation. (WWW, 2022) [[paper](https://arxiv.org/abs/2204.00926)]
- A Study of Defensive Methods to Protect Visual Recommendation Against Adversarial Manipulation of Images. (SIGIR, 2021) [[paper](https://dl.acm.org/doi/10.1145/3404835.3462848)]
- Attacking Recommender Systems with Augmented User Profiles. (CIKM, 2020) [[paper](https://dl.acm.org/doi/10.1145/3340531.3411884https://dl.acm.org/doi/10.1145/3340531.3411884)]
- Practical Data Poisoning Attack against Next-Item Recommendation. (WWW, 2020) [[paper](http://arxiv.org/abs/2004.03728)]
- Adversarial attacks on an oblivious recommender. (RecSys, 2019) [[paper](https://dl.acm.org/doi/epdf/10.1145/3298689.3347031)]
- Adversarial Collaborative Auto-encoder for Top-N Recommendation. (IJCNN, 2019) [[paper](http://arxiv.org/abs/1808.05361)]
- Adversarial Personalized Ranking for Recommendation. (SIGIR, 2018) [[paper](http://arxiv.org/abs/1808.03908)] [[code](https://github.com/hexiangnan/adversarial_personalized_ranking)]
- Data Poisoning Attacks on Factorization-Based Collaborative Filtering. (NIPS, 2016) [[paper](https://arxiv.org/abs/1608.08182)] [[code](https://github.com/fuying-wang/Data-poisoning-attacks-on-factorization-based-collaborative-filtering)]
- Catch the Black Sheep: Unified Framework for Shilling Attack Detection Based on Fraudulent Action Propagation. (IJCAI, 2015) [[paper](https://www.ijcai.org/Proceedings/15/Papers/341.pdf)] 


## Graph


- How Powerful is Graph Convolution for Recommendation? (CIKM, 2021) [[paper](http://arxiv.org/abs/2108.07567)] [[code](https://github.com/yshenaw/GF_CF)]
- Minimizing Polarization and Disagreement in Social Networks via Link Recommendation. (NIPS, 2021) [[paper](https://papers.nips.cc/paper/2021/file/101951fe7ebe7bd8c77d14f75746b4bc-Paper.pdf)]
- LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. (SIGIR, 2020) [[paper](http://arxiv.org/abs/2002.02126)] [[PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)] [[TensorFlow](https://github.com/kuandeng/LightGCN)]



## Seminal

- Variational Autoencoders for Collaborative Filtering. (WWW, 2018) [[paper](https://www.cnblogs.com/MTandHJ/p/16460617.html)] [[code](https://github.com/dawenl/vae_cf)]
- Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks. (IJCAI, 2017) [[paper](https://www.ijcai.org/proceedings/2017/435)] [[code](https://github.com/hexiangnan/attentional_factorization_machine)] [[PyTorch](https://github.com/shenweichen/DeepCTR-PyTorch)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- Neural Factorization Machines for Sparse Predictive Analytics. (SIGIR, 2017) [[paper](https://dl.acm.org/doi/10.1145/3077136.3080777)] [[code](https://github.com/hexiangnan/neural_factorization_machine)] [[PyTorch](https://github.com/xue-pai/FuxiCTR)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. (IJCAI, 2017) [[paper](https://www.ijcai.org/Proceedings/2017/0239.pdf)] [[code](https://github.com/xue-pai/FuxiCTR)] [[PyTorch](https://github.com/shenweichen/DeepCTR-PyTorch)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- Deep Interest Network for Click-Through Rate Prediction. (SIGKDD, 2018) [[paper](http://arxiv.org/abs/1706.06978)]
- Deep & Cross Network for Ad Click Predictions. (AdKDD, 2017) [[paper](http://arxiv.org/abs/1708.05123)] [[PyTorch](https://github.com/shenweichen/DeepCTR-Torch)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- Wide & Deep Learning for Recommender Systems. (2016) [[paper](https://arxiv.org/abs/1606.07792)] [[PyTorch](code)] [[TensorFlow](https://github.com/microsoft/recommenders)]
- Field-aware Factorization Machines for CTR Prediction. (RecSys, 2016) [[paper](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)]
- Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features. (SIGKDD, 2016) [[paper](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)] [[code](https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/models/DeepCrossing.py)]
- Factorization Machines. (ICDM, 2010) [[paper](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)]
- AutoRec: Autoencoders Meet Collaborative Filtering. (WWW, 2015) [[paper](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)] [[code](https://github.com/NeWnIx5991/AutoRec-for-CF)]
- Amazon.com Recommendations Item-to-Item Collaborative Filtering. (IEEE Internet Computing, 2003) [[code](https://github.com/rita05616/Amazon-Recommendation-System)]



