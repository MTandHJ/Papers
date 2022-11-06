





## Fairness

- Rank List Sensitivity of Recommender Systems to Interaction Perturbations. (CIKM) [[paper](http://arxiv.org/abs/2201.12686)]
- Towards Principled User-side Recommender Systems. (CIKM, 2022) [[paper](http://arxiv.org/abs/2208.09864)]
- Unbiased Learning to Rank with Biased Continuous Feedback. (CIKM, 2022) [[paper](https://dl.acm.org/doi/10.1145/3511808.3557483)] [[code](https://github.com/phyllist/ULTRA)] (novel; empirical; position bias; trust bias; IPS)
- Representation Matters When Learning From Biased Feedback in Recommendation. (CIKM, 2022) [[paper](https://dl.acm.org/doi/10.1145/3511808.3557431)] (novel; theoretical; selection bias; adversarial; divergence)
- Quantifying and Mitigating Popularity Bias in Conversational Recommender Systems. (CIKM, 2022) [[paper](http://arxiv.org/abs/2208.03298)] (emmm; empirical; CTR; popularity bias)
- EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks. (WWW, 2022) [[paper](https://www.cnblogs.com/MTandHJ/p/16772522.html)] [[code](https://github.com/yushundong/EDITS)] (wow; seminal; empirical; graph; GNN; fairness)
- Invariant Preference Learning for General Debiasing in Recommendation. (KDD, 2022) [[paper](https://dl.acm.org/doi/10.1145/3534678.3539439)] [[code](https://github.com/AIflowerQ/InvPref_KDD_2022)] (debias; EM; adversarial training)
- Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis. (KDD, 2022) [[paper](https://dl.acm.org/doi/10.1145/3534678.3539240)] [[code](https://github.com/Dingseewhole/Robust_Deconfounder_master/)] (selection bias; novel)
- Comprehensive Fair Meta-learned Recommender System. (KDD, 2022) [[paper](http://arxiv.org/abs/2206.04789)] (Fairness; 元学习; 多任务)
- Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. (SIGIR, 2022) [[paper](https://dl.acm.org/doi/epdf/10.1145/3477495.3531889)]
- Explainable Fairness in Recommendation. (SIGIR, 2022) [[paper](http://arxiv.org/abs/2204.11159)]
- Fairness in Recommendation: A Survey. [[paper](http://arxiv.org/abs/2205.13619)] 对 Fairness 的介绍挺详尽的
- CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems. (SIGIR, 2022) [[paper](https://arxiv.org/abs/2204.08085)]
- Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information. (WSDM, 2021) [[paper](https://dl.acm.org/doi/10.1145/3437963.3441752)] (wow; theoretical; graph; GNN; fairness)
- AutoDebias: Learning to Debias for Recommendation. (SIGIR, 2021) [[paper](http://arxiv.org/abs/2105.04170)] [[code](https://github.com/DongHande/AutoDebias)]
- Towards Long-term Fairness in Recommendation. (WSDM, 2021) [[paper](http://arxiv.org/abs/2101.03584)]
- Fairness among New Items in Cold Start Recommender Systems. (SIGIR, 2021) [[paper](https://dl.acm.org/doi/10.1145/3404835.3462948)]
- Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System. (KDD, 2021) [[paper](http://arxiv.org/abs/2010.15363)]
- FairRec: Fairness-aware News Recommendation with Decomposed Adversarial Learning. (AAAI, 2021) [[paper](https://arxiv.org/abs/2006.16742)]
- Enhanced Doubly Robust Learning for Debiasing Post-click Conversion Rate Estimation. (SIGIR, 2021) [[paper](http://arxiv.org/abs/2105.13623)] [[code](https://github.com/guosyjlu/MRDR-DL)] (Double robustness; Post-click; IPS; MNAR)
- Fairness without Demographics through Adversarially Reweighted Learning. (NIPS, 2020) [[paper](http://arxiv.org/abs/2006.13114)] [[code](https://github.com/google-research/google-research/tree/master/group_agnostic_fairness)] (novel; empirical; fairness)
- A Framework for Understanding Sources of Harm throughout the Machine Learning Life. [[paper](http://arxiv.org/abs/1901.10002)]
- Multi-stakeholder Recommendation and its Connection to Multi-sided Fairness. (RecSys, 2019) [[paper](http://arxiv.org/abs/1907.13158)]
- Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random. (ICML, 2019) [[paper](http://yusun-aldrich.com/paper/dr_mnar_full.pdf)]
- Unbiased offline recommender evaluation for missing-not-at-random implicit feedback. (RecSys, 2018) [[paper](https://vision.cornell.edu/se3/wp-content/uploads/2018/08/recsys18_unbiased_eval.pdf)]
- Causal Embeddings for Recommendation. (RecSys 2018) [[paper](http://arxiv.org/abs/1706.07639)] [[code](https://github.com/criteo-research/CausE)] (selection bias; casual; uniform)
- Beyond Parity: Fairness Objectives for Collaborative Filtering. (NIPS, 2017) [[paper](http://arxiv.org/abs/1705.08804)]  提出了几种 Fairness 指标, 可以直接作为一般训练方法的正则项 (文中基于最普通的协同过滤).
- Counterfactual Fairness. (NIPS, 2017) [[paper](https://papers.nips.cc/paper/2017/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html)] 定义了一种 Counterfactual Fairness, 但是除此之外没看出亮点.
- Multisided Fairness for Recommendation. (FATML, 2017) [[paper](http://arxiv.org/abs/1707.00093)] 对多方面的 Fairness 进行一个介绍
- Fairness-Aware Group Recommendation with Pareto-Efficiency. (RecSys, 2017) [[paper](https://dl.acm.org/doi/10.1145/3109859.3109887)]
- Ranking with Fairness Constraints. [[paper](http://arxiv.org/abs/1704.06840)] 本文讨论在一种'强硬'的 Fairness 约束下, 如何 (快速) re-ranking 以保留尽可能多的收益.
- Recommendations as Treatments: Debiasing Learning and Evaluation. (ICML, 2016) [[paper](https://www.cs.cornell.edu/people/tj/publications/schnabel_etal_16b.pdf)] [[code](https://www.cs.cornell.edu/~schnabts/mnar/)]
- Causal Inference for Recommendation. (2016) [[paper](https://dawenl.github.io/publications/LiangCB16-causalrec.pdf)] (selection bias; causal; '无偏'数据集采集方法)
- Evaluation of recommendations: rating-prediction and ranking. (RecSys, 2013) [[paper](https://dl.acm.org/doi/10.1145/2507157.2507160)] 利用插值方法处理推荐系统中的缺失数据, 对三种状况的分析还算有趣.


## Diversity

- Neural Re-ranking in Multi-stage Recommender Systems: A Review. (arXiv preprint arXiv:2202.06602) [[paper](http://arxiv.org/abs/2202.06602)] [[code](https://github.com/LibRerank-Community/LibRerank)]
- Rabbit Holes and Taste Distortion: Distribution-Aware Recommendation with Evolving Interests. (WWW, 2021) [[paper](https://people.engr.tamu.edu/caverlee/pubs/WWW21-Final-Publication.pdf)]

## Privacy

- FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation. (ICML, workshop, 2021) [[paper](https://arxiv.org/abs/2102.04925v2)] (novel; graph; GNN; privacy; federated learning)
- Graph Embedding for Recommendation against Attribute Inference Attacks. (WWW, 2021) [[paper](http://arxiv.org/abs/2101.12549)]


## Explainability


- ProtGNN: Towards Self-Explaining Graph Neural Networks. (AAAI, 2022) [[paper](http://arxiv.org/abs/2112.00911)] (emmm; graph; empirical; explainability; GNN)
- Causal screening to interpret graph neural networks. (2021) (emmm; empriical; casual; explainability; graph; GNN)
- XGNN: Towards Model-Level Explanations of Graph Neural Networks. (KDD, 2020) [[paper](http://arxiv.org/abs/2006.02587)] (emmm; graph; empirical; RL; explainability; GNN; model-level; graph-level)
- GNNExplainer: Generating Explanations for Graph Neural Networks. (NIPS, 2019) [[paper](http://arxiv.org/abs/1903.03894)] [[code](https://github.com/RexYing/gnn-model-explainer)] (emmm; empirical; graph; GNN; post hoc; explainability)
- Explainability Methods for Graph Convolutional Neural Networks. (CVPR, 2019) [[paper](https://ieeexplore.ieee.org/document/8954227/)] (emmmm; empirical; CAM; EB; graph; GNN; explainability)

## Cold-start

- Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation. (WSDM, 2021) [[paper](http://arxiv.org/abs/2012.07064)] (emmm; empirical; graph; GNN; pretraining; cold-start)

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

- MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering. (2022) [[paper](http://arxiv.org/abs/2204.02338)] [[code](https://github.com/hujunxianligong/MGDCF)] (emmm; empirical; graph; GNN; diffusion, markov)
- Collaboration-Aware Graph Convolutional Network for Recommender Systems. (2022) [[paper](http://arxiv.org/abs/2207.06221)] [[code](https://github.com/YuWVandy/CAGCN)] (novel; wow; theoretical; graph; GNN; reweight)
- SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation. (CIKM, 2022) [[paper](http://arxiv.org/abs/2208.12689)] [[code](https://github.com/tanatosuu/svd_gcn)] (novel; theoretical; graph; GNN; over-smoothing)
- Localized Graph Collaborative Filtering. (2022) [[paper](http://arxiv.org/abs/2108.04475)] (舍弃 embeddings)
- Research on Recommendation Algorithm of Joint Light Graph Convolution Network and DropEdge. (Journal of Advanced Transportation, 2022)
- Towards Representation Alignment and Uniformity in Collaborative Filtering. (KDD, 2022) [[paper](http://arxiv.org/abs/2206.12811)] [[code](https://github.com/THUwangcy/DirectAU)]
- Knowledge Graph Contrastive Learning for Recommendation. (SIGIR, 2022) [[paper](http://arxiv.org/abs/2205.00976)] [[code](https://github.com/yuh-yang/KGCL-SIGIR22)] (图 + 知识图谱)
- Less is More: Reweighting Important Spectral Graph Features for Recommendation. (SIGIR, 2022) [[paper](http://arxiv.org/abs/2204.11346)] [[code](https://github.com/tanatosuu/GDE)] (关于 over-smoothing 的分析倒是有趣)
- A Review-aware Graph Contrastive Learning Framework for Recommendation. (SIGIR, 2022) [[paper](http://arxiv.org/abs/2204.12063)]
- UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation. (CIKM, 2021) [[paper](http://arxiv.org/abs/2110.15114)] [[code](https://github.com/xue-pai/UltraGCN)] (novel; theoretical; over-smoothing; graph; GNN)
- SimpleX: A Simple and Strong Baseline for Collaborative Filtering. (CIKM, 2021) [[paper](http://arxiv.org/abs/2109.12613)] [[code](https://github.com/xue-pai/TwoTowers)] (TwoTowers)
- How Powerful is Graph Convolution for Recommendation? (CIKM, 2021) [[paper](http://arxiv.org/abs/2108.07567)] [[code](https://github.com/yshenaw/GF_CF)]
- Minimizing Polarization and Disagreement in Social Networks via Link Recommendation. (NIPS, 2021) [[paper](https://papers.nips.cc/paper/2021/file/101951fe7ebe7bd8c77d14f75746b4bc-Paper.pdf)]
- Deoscillated Graph Collaborative Filtering. (2020) [[paper](http://arxiv.org/abs/2011.02100)] [[code](https://www.cnblogs.com/MTandHJ/p/16841462.html)] (novel; empirical; graph; GNN)
- Learning to Hash with Graph Neural Networks for Recommender Systems. (WWW, 2020) [[paper](https://dl.acm.org/doi/10.1145/3366423.3380266)] (emmm; empirical; graph; GNN; hash; recall)
- Hierarchical Bipartite Graph Neural Networks: Towards Large-Scale E-commerce Applications. (ICDE, 2020) [[paper](https://ieeexplore.ieee.org/document/9101846/)] (novel; empirical; sampling; large-scale)
- LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. (SIGIR, 2020) [[paper](http://arxiv.org/abs/2002.02126)] [[PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)] [[TensorFlow](https://github.com/kuandeng/LightGCN)]
- Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach. (AAAI, 2020) [[paper](http://arxiv.org/abs/2001.10167)] [[code](https://github.com/newlei/LR-GCCF)] (emmm; empirical; over-smoothing; graph; GNN)
- Joint Item Recommendation and Attribute Inference: An Adaptive Graph Convolutional Network Approach. (SIGIR, 2020) [[paper](http://arxiv.org/abs/2005.12021)] (emmm; empirical; graph; GNN; joint)
- STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems. (IJCAI, 2019) [[paper](http://arxiv.org/abs/1905.13129)] [[code](https://github.com/jennyzhang0215/STAR-GCN)] (emmm; empirical; cold-start; graph; GNN)
- Neural Graph Collaborative Filtering. (SIGIR, 2019) [[paper](http://arxiv.org/abs/1905.08108)] (NGCF)
- Graph Convolutional Neural Networks for Web-Scale Recommender Systems. (KDD, 2018) [[paper](https://arxiv.org/abs/1806.01973)] (PinSage; large-scale; novel; empirical; random-walk)
- HOP-rec: high-order proximity for implicit recommendation. (RecSys, 2018) (Hop-Rec, 分级处理高阶信息)
- Graph Convolutional Matrix Completion. (KDD, 2017) (GCMC) [[paper](https://arxiv.org/abs/1706.02263)] [[code](https://github.com/riannevdberg/gc-mc)] [[PyTorch](https://github.com/hengruizhang98/GCMC-Pytorch-dgl)]
- Random-Walk Computation of Similarities between Nodes of a Graph with Application to Collaborative Recommendation. (TKDE, 2017) [[paper](https://www.researchgate.net/publication/3297672_Random-Walk_Computation_of_Similarities_between_Nodes_of_a_Graph_with_Application_to_Collaborative_Recommendation)] (介绍 first-passage time / cost 以及相应的 CTD (Commute Time Distance) 的显式计算公式)
- DeepWalk: Online Learning of Social Representations. (KDD, 2014) [[paper](http://arxiv.org/abs/1403.6652)]
- Supervised Random Walks: Predicting and Recommending Links in Social Networks. (WSDM, 2011) [[paper](https://arxiv.org/abs/1011.4071)] [[code](https://github.com/Pepton21/Supervised-random-walks-with-restarts)] (graph; PageRank; random walk)
- ItemRank: A Random-Walk Based Scoring Algorithm for Recommender Engines. (IJCAI, 2007) [[paper](https://www.ijcai.org/Proceedings/07/Papers/444.pdf)]
- Fast Random Walk with Restart and Its Applications. (ICDM, 2006) [[paper](http://ieeexplore.ieee.org/document/4053087/)]

## Seminal

- ItemSage: Learning Product Embeddings for Shopping Recommendations at Pinterest. (KDD, 2022) [[paper](http://arxiv.org/abs/2205.11728)] (novel; empirical; seminal; multi-task)
- Variational Autoencoders for Collaborative Filtering. (WWW, 2018) [[paper](https://www.cnblogs.com/MTandHJ/p/16460617.html)] [[code](https://github.com/dawenl/vae_cf)]
- Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks. (IJCAI, 2017) [[paper](https://www.ijcai.org/proceedings/2017/435)] [[code](https://github.com/hexiangnan/attentional_factorization_machine)] [[PyTorch](https://github.com/shenweichen/DeepCTR-PyTorch)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- Neural Factorization Machines for Sparse Predictive Analytics. (SIGIR, 2017) [[paper](https://dl.acm.org/doi/10.1145/3077136.3080777)] [[code](https://github.com/hexiangnan/neural_factorization_machine)] [[PyTorch](https://github.com/xue-pai/FuxiCTR)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. (IJCAI, 2017) [[paper](https://www.ijcai.org/Proceedings/2017/0239.pdf)] [[code](https://github.com/xue-pai/FuxiCTR)] [[PyTorch](https://github.com/shenweichen/DeepCTR-PyTorch)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- Deep Interest Network for Click-Through Rate Prediction. (KDD, 2018) [[paper](http://arxiv.org/abs/1706.06978)]
- Deep & Cross Network for Ad Click Predictions. (AdKDD, 2017) [[paper](http://arxiv.org/abs/1708.05123)] [[PyTorch](https://github.com/shenweichen/DeepCTR-Torch)] [[TensorFlow](https://github.com/shenweichen/DeepCTR)]
- Wide & Deep Learning for Recommender Systems. (2016) [[paper](https://arxiv.org/abs/1606.07792)] [[PyTorch](code)] [[TensorFlow](https://github.com/microsoft/recommenders)]
- Field-aware Factorization Machines for CTR Prediction. (RecSys, 2016) [[paper](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)]
- Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features. (KDD, 2016) [[paper](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)] [[code](https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/models/DeepCrossing.py)]
- Factorization Machines. (ICDM, 2010) [[paper](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)]
- AutoRec: Autoencoders Meet Collaborative Filtering. (WWW, 2015) [[paper](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)] [[code](https://github.com/NeWnIx5991/AutoRec-for-CF)]
- Amazon.com Recommendations Item-to-Item Collaborative Filtering. (IEEE Internet Computing, 2003) [[code](https://github.com/rita05616/Amazon-Recommendation-System)]
- The PageRank Citation Ranking: Bringing Order to the Web. (Technical report, 1998) [[paper](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)]


## Survey

- A Survey on Trustworthy Recommender Systems. [[paper](http://arxiv.org/abs/2207.12515)]


## Other

- Cache-Augmented Inbatch Importance Resampling for Training Recommender Retriever. (NIPS, 2022) [[paper](http://arxiv.org/abs/2205.14859)] (importance sampling; theoretical; novel)
- Adversarial Gradient Driven Exploration for Deep Click-Through Rate Prediction. (KDD, 2022) [[paper](http://arxiv.org/abs/2112.11136)] (novel; theoretical; E&E; adversarial)
- Improving Location Recommendation with Urban Knowledge Graph. (KDD, 2022) [[paper](http://arxiv.org/abs/2111.01013)] (POI; graph; casual inference)
- DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation. (SIGIR, 2022) [[paper](https://dl.acm.org/doi/10.1145/3477495.3531967)] [[code](https://github.com/cjx96/DisenCDR)] (跨域训练, 互信息)
- Explainable Recommendation with Comparative Constraints on Product Aspects. (WSDM, 2021) [[paper](https://dl.acm.org/doi/10.1145/3437963.3441754)] (可解释性)
- Personalized Ranking with Importance Sampling. (WWW, 2020) [[paper](https://dl.acm.org/doi/10.1145/3366423.3380187)] (importance sampling; theoretical; novel)
- Sampling-bias-corrected neural modeling for large corpus item recommendations. [[paper](https://dl.acm.org/doi/10.1145/3298689.3346996)]  (importance sampling; theoretical; novel)
- Sampling-bias-corrected neural modeling for large corpus item recommendations. (RecSys, 2019) [[paper](https://dl.acm.org/doi/10.1145/3298689.3346996)] (novel; empirical; bias; importance sampling)
- Adaptive Sampled Softmax with Kernel Based Sampling. (ICML, 2018) [[paper](http://arxiv.org/abs/1712.00527)] (importance sampling; theoretical; novel)
- Putting Users in Control of their Recommendations. (RecSys, 2015) [[paper](https://dl.acm.org/doi/10.1145/2792838.2800179)] (交互, 用户可控)
- Collaborative Filtering for Implicit Feedback Datasets. (ICDM, 2008) [[paper](http://yifanhu.net/PUB/cf.pdf)] (MF 在 implicit data 上的应用)
- Being Accurate is Not Enough: How Accuracy Metrics have hurt Recommender Systems. (2006) (测度)

