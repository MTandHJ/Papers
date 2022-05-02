
# Robust Learning


## Optimization

修改损失, 学习策略等避免局部最优的方法.

- Enhancing Adversarial Training with Second-Order Statistics of Weights. (CVPR, 2022) [[paper](https://arxiv.org/abs/2203.06020)] [[code](https://github.com/Alexkael/S2O)]
- Double Descent in Adversarial Training: An Implicit Label Noise Perspective. (ICLR, 2022) [[paper](http://arxiv.org/abs/2110.03135)]
- Exploring Memorization in Adversarial Training. (ICLR, 2022) [[paper](http://arxiv.org/abs/2106.01606)] [[code](https://github.com/dongyp13/memorization-AT)]
- Towards the Memorization Effect of Neural Networks in Adversarial Training. (ICLR, 2022) [[paper](http://arxiv.org/abs/2106.04794)] [[code](https://anonymous.4open.science/r/benign-adv-77C5)]
- DropAttack: A Masked Weight Adversarial Training Method to Improve Generalization of Neural Networks. (ICLR, 2022) [[paper](http://arxiv.org/abs/2108.12805)] [[code](https://github.com/nishiwen1214/DropAttack)]
- Fixing Data Augmentation to Improve Adversarial Robustness. (2021) [[paper](https://arxiv.org/abs/2103.01946?msclkid=655be8b5b19111ec99add64c2edadd77)]
- Bag of Tricks for Adversarial Training. (ICLR, 2021) [[paper](http://arxiv.org/abs/2010.00467)] [[code](https://github.com/P2333/Bag-of-Tricks-for-AT?msclkid=c5080f60b19311eca2e42624e9ecec78)]
- Helper-Based Adversarial Training: Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-Off. (ICML, 2021) [[paper](https://openreview.net/pdf?id=BuD2LmNaU3a)] [[code](https://github.com/imrahulr/hat)]
- Improving Adversarial Robustness Using Proxy Distributions. (2021) [[paper](https://arxiv.org/abs/2104.09425v1)]
- Adversarial Weight Perturbation Helps Robust Generalization. (NIPS, 2020) [[paper]](http://arxiv.org/abs/2004.05884) [[code]](https://github.com/csdongxian/AWP)
- Fast is Better than Free: Revisiting Adversarial Training. (ICLR, 2020) [[paper](http://arxiv.org/abs/2001.03994)] [[code](https://github.com/locuslab/fast_adversarial)]
- Attacks Which Do Not Kill Training Make Adversarial Learning Stronger. (ICML, 2020) [[paper](http://arxiv.org/abs/2002.11242)] [[code](https://github.com/zjfheart/Friendly-Adversarial-Training)]
- Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization. (CVPR, 2020) [[paper](https://arxiv.org/abs/2003.02484v3#:~:text=Adversarial%20Vertex%20mixup%20%28AVmixup%29%2C%20a%20soft-labeled%20data%20augmentation,and%20show%20that%20AVmixup%20significantly%20improves%20the%20robust)] [[code](https://arxiv.org/abs/2003.02484v3#:~:text=Adversarial%20Vertex%20mixup%20%28AVmixup%29%2C%20a%20soft-labeled%20data%20augmentation,and%20show%20that%20AVmixup%20significantly%20improves%20the%20robust)] [[PyTorch](https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch)]
- Understanding and Improving Fast Adversarial Training. (NIPS, 2020) [[paper](https://arxiv.org/abs/2007.02617)] [[code](https://github.com/tml-epfl/understanding-fast-adv-training)]
- Robust Pre-Training by Adversarial Contrastive Learning. (NIPS, 2020) [[paper](https://arxiv.org/abs/2010.13337)] [[code](https://github.com/VITA-Group/Adversarial-Contrastive-Learning)]
- Adversarial Self-Supervised Contrastive Learning. (NIPS, 2020) [[paper](https://arxiv.org/abs/2006.07589)] [[code](https://github.com/Kim-Minseon/RoCL)]
- Boosting Adversarial Training with Hypersphere Embedding. (NIPS, 2020) [[paper](http://arxiv.org/abs/2002.08619)] [[code](https://github.com/ShawnXYang/AT_HE)]
- **Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples.** (2020) [[paper](https://arxiv.org/abs/2010.03593)]
- Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness. (ICLR, 2020) [[paper](http://arxiv.org/abs/1905.10626)] [[code](https://github.com/P2333/Max-Mahalanobis-Training)]
- Second Order Optimization for Adversarial Robustness and Interpretability. (2020) [[paper](https://arxiv.org/pdf/2009.04923.pdf)]
- Improving Adversarial Robustness Requires Revisiting Misclassified Examples. (ICLR, 2020) [[paper](https://openreview.net/forum?id=rklOg6EFwS)] [[code](https://github.com/YisenWang/MART?msclkid=8ee60abdb18b11ec88297eb510e53f39)]
- Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks. (ICCV, 2019) [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mustafa_Adversarial_Defense_by_Restricting_the_Hidden_Space_of_Deep_Neural_ICCV_2019_paper.pdf)]
- **Theoretically Principled Trade-off between Robustness and Accuracy.** (ICML, 2019) [[paper](http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf)] [[code](https://github.com/yaodongyu/TRADES)]
- Adversarial Training for Free! (NIPS, 2019) [[paper](https://papers.nips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf?msclkid=b49f3cffb19211ec9eba4b8e1378fe4b)]
- Adversarial Logit Pairing. (2018) [[paper](http://arxiv.org/abs/1803.06373)]
- **Towards Deep Learning Models Resistant to Adversarial Attacks.** (ICLR, 2018) [[paper](https://arxiv.org/pdf/1706.06083.pdf)] [[code](https://github.com/MadryLab/cifar10_challenge)]


## Detection

检测方法大抵通过给定阈值判断可靠性.


- Adversarial Training with Rectified Rejection. (CVPR, 2022) [[paper](https://arxiv.org/pdf/2105.14785.pdf)] [[code](https://github.com/P2333/Rectified-Rejection)]
- Confidence-Calibrated Adversarial Training: Generalizing to Unseen Attacks. (ICML, 2020) [[paper](http://arxiv.org/abs/1910.06259)] [[code](https://github.com/davidstutz/confidence-calibrated-adversarial-training)]
- Energy-Based Out-of-Distribution Detection. (NIPS, 2020) [[paper](https://arxiv.org/abs/2010.03759v4)]
- ATRO: Adversarial Training with A Rejection Option. (2020) [[paper](https://arxiv.org/abs/2010.12905#:~:text=Adversarial%20training%20is%20one%20of%20them%2C%20which%20trains,of%20Adversarial%20Training%20with%20a%20Rejection%20Option%20%28ATRO%29.)]
- Playing It Safe: Adversarial Robustness with An Abstain Option. (2019) [[paper](https://arxiv.org/abs/1911.11253)] [[code](https://github.com/cassidylaidlaw/playing-it-safe)]
- SelectiveNet: A Deep Neural Network with An Integrated Reject Option. (ICML, 2019) [[paper](http://arxiv.org/abs/1901.09192)] [[code](https://github.com/geifmany/SelectiveNet)]
- Robust Detection of Adversarial Attacks by Modeling the Intrinsic Properties of Deep Neural Networks. (NIPS, 2018) [[paper](https://papers.nips.cc/paper/2018/file/e7a425c6ece20cbc9056f98699b53c6f-Paper.pdf)]
- A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. (NIPS, 2018) [[paper](https://arxiv.org/abs/1807.03888v2)] [[code](https://github.com/pokaxpoka/deep_Mahalanobis_detector)]
- Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality. (ICLR, 2018) [[paper](https://arxiv.org/abs/1801.02613)]
- Detecting Adversarial Samples from Artifacts. (2017) [[paper](https://arxiv.org/abs/1703.00410)] [[code](https://github.com/rfeinman/detecting-adversarial-samples)]

## Certified

可验证的防御方法.

- Globally-Robust Neural Networks. (ICML, 2021) [[paper](http://arxiv.org/abs/2102.08452)] [[code](https://github.com/klasleino/gloro/blob/master/gloro/models.py)]
- CROWN-IBP: Towards Stable and Efficient Training of Verifiably Robust Neural Networks. (ICLR, 2020) [[paper](https://arxiv.org/abs/1906.06316)] [[code](https://github.com/huanzhang12/CROWN-IBP)]
- Certified Adversarial Robustness via Randomized Smoothing. (ICML, 2019) [[paper](https://arxiv.org/pdf/1902.02918.pdf)] [[code](https://github.com/locuslab/smoothing)]
- Certified Robustness to Adversarial Examples with Differential Privacy. (S&P, 2019) [[paper](https://arxiv.org/abs/1802.03471)]
- Scalable verified training for provably robust image classification. (ICCV, 2019) [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gowal_Scalable_Verified_Training_for_Provably_Robust_Image_Classification_ICCV_2019_paper.pdf)]
- Efficient Neural Network Robustness Certification with General Activation Functions. (NIPS, 2018) [[paper](https://arxiv.org/abs/1811.00866)] [[code](https://github.com/deepmind/interval-bound-propagation)]


## Architecture

网络结构因素.

- Parameterizing Activation Functions for Adversarial Robustness. (2022) [[paper]](https://arxiv.org/pdf/2110.05626v1.pdf)
- Do Wider Neural Networks Really Help Adversarial Robustness? (NIPS, 2021) [[paper](http://arxiv.org/abs/2010.01279)]
- Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks. (NIPS, 2021) [[paper](http://arxiv.org/abs/2110.03825)] [[code](https://github.com/HanxunH/RobustWRN)]
- On the Robustness of Vision Transformers to Adversarial Examples. (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.pdf?msclkid=3dc37010b19211ec9808c62953b9d693)]
- Understanding Robustness of Transformers for Image Classification. (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Bhojanapalli_Understanding_Robustness_of_Transformers_for_Image_Classification_ICCV_2021_paper.pdf?msclkid=1b733cf2b19211ec85bbc10d4a9a4d95)]
- Improving Adversarial Robustness via Channel-Wise Activation Suppressing. (ICLR, 2021) [[paper](https://arxiv.org/pdf/2103.08307.pdf)] [[code](https://github.com/bymavis/CAS_ICLR2021)]
- Improving Adversarial Robustness of CNNs via Channel-Wise Importance-Based Feature Selection. (ICML, 2021) [[paper](http://arxiv.org/abs/2102.05311)]



## Heuristic

- Adversarial Examples Improve Image Recognition. (CVPR, 2019) [[paper](https://arxiv.org/abs/1911.09665)]
- Defending Adversarial Attacks by Correcting Logits. (2019) [[paper](https://arxiv.org/pdf/1906.10973v1.pdf)]
- Max-Mahalanobis Linear Discriminant Analysis Networks. (ICML, 2018) [[paper](https://arxiv.org/abs/1802.09308)]
- A New Defense Against Adversarial Images: Turning a Weakness into a Strength. (NIPS, 2019) [[paper]()] [[code]()]
- Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks. (S&P, 2016) [[paper](https://arxiv.org/pdf/1511.04508.pdf)]

## Theory

- What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. (NIPS, 2020) [[paper](https://arxiv.org/abs/2008.03703)]
- Does Learning Require Memorization? A Short Tale about a Long Tail. (2020) [[paper](http://arxiv.org/abs/1906.05271)]
- **Adversarial Examples Are Not Bugs, They Are Features.** (NIPS, 2019) [[paper](https://arxiv.org/pdf/1905.02175.pdf)]
- **Adversarially Robust Generalization Requires More Data.** (NIPS, 2018) [[paper](https://arxiv.org/abs/1804.11285)]
- Understanding Black-box Predictions via Influence Functions. (ICML, 2017) [[paper]](https://arxiv.org/pdf/1703.04730.pdf)




## Attack


- Mind the Box: ℓ1-APGD for Sparse Adversarial Attacks on Image Classifiers. (ICML, 2021) [[paper](https://arxiv.org/abs/2103.01208)] [[code](https://github.com/fra31/auto-attack)]
- **Reliable Evaluation of Adversarial Robustness with An Ensemble of Diverse Parameter-Free Attacks.** (ICML, 2020) [[paper](http://arxiv.org/abs/2003.01690)] [[code](https://github.com/fra31/auto-attack)]
- AT-GAN: A Generative Attack Model for Adversarial Transferring on Generative Adversarial Nets. (2019) [[paper](https://arxiv.org/abs/1904.07793)]
- One Pixel Attack for Fooling Deep Neural Networks. (2019) [[paper](https://arxiv.org/pdf/1710.08864.pdf)]
- Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. (ICML, 2018) [[paper](https://arxiv.org/pdf/1802.00420.pdf)]
- Universal Adversarial Perturbations. (CVPR, 2017) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf)] [[code](https://github.com/LTS4/universal)]
- DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. (CVPR, 2016) [[paper](https://arxiv.org/pdf/1511.04599.pdf)]
- **Towards Evaluating the Robustness of Neural Networks.** (S&P, 2017) [[paper](https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf)]
- Adversarial Examples In The Physical World. (2016) [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45471.pdf)]
- The Limitations of Deep Learning in Adversarial Settings. (2015) [[paper](https://arxiv.org/pdf/1511.07528.pdf)]
- **Practical Black-Box Attacks against Machine Learning.** (2017) [[paper](https://arxiv.org/pdf/1602.02697.pdf)]
- **Explaining and Harnessing Adversarial Examples.** (ICLR, 2015) [[paper](https://arxiv.org/abs/1412.6572)]
- **Intriguing Properties of Neural Networks.** (ICLR, 2014) [[paper](http://arxiv.org/abs/1312.6199)]

## Generalization

对于语义不变变换的鲁棒性.


- The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization. (ICCV, 2020) [[paper](http://arxiv.org/abs/2006.16241)] [[code](https://github.com/hendrycks/imagenet-r)]
- **Augmix : A Simple Data Processing Method to Improve Robustness and Uncertainty.** (ICLR, 2020) [[paper](https://openreview.net/pdf?id=S1gmrxHFvB)] [[code](https://github.com/google-research/augmix?msclkid=8ceb09b3b18d11ecb4200eb4bfedb6cf)]
- RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.[[paper](https://arxiv.org/abs/1909.13719v2)]
- AutoAugment: Learning Augmentation Strategies from Data. (CVPR, 2019) [[paper](http://export.arxiv.org/pdf/1805.09501)]
- Sharpness-Aware Minimization for Efficiently Improving Generalization. (ICLR, 2020) [[paper](https://arxiv.org/abs/2010.01412v3)] [[code](https://github.com/davda54/sam)]
- Cutmix: Regularization Strategy to Train Strong Classifiers with Localizable Features. (ICCV, 2019) [[paper](https://arxiv.org/abs/1905.04899v2)] [[code](https://github.com/clovaai/CutMix-PyTorch)]
- Mixup: Beyond Empirical Risk Minimization. (ICLR, 2018) [[paper](http://arxiv.org/abs/1710.09412)] [[code](https://github.com/facebookresearch/mixup-cifar10)]
- Improved Regularization of Convolutional Neural Networks with Cutout. (2017) [[paper](https://arxiv.org/abs/1708.04552)] [[code](https://github.com/uoguelph-mlrg/Cutout)]
- Deep Residual Learning for Image Recognition. (2016) [[paper](https://arxiv.org/abs/1512.03385?source=post_page---------------------------)]