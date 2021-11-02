
# GCE NN
This repository contains the `Tensorflow` implementation of the papers
* *Dim but not entirely dark: Extracting the Galactic Center Excess' source-count distribution with neural nets*, Florian List, Nicholas L. Rodd, and Geraint F. Lewis,  accepted by *Phys. Rev. D*, 2021 [[arXiv:2107.09070](https://arxiv.org/abs/2107.09070)].
 **→** Branch "**prd**"  
* *Galactic Center Excess in a New Light: Disentangling the γ-Ray Sky with Bayesian Graph Convolutional Neural Networks*, Florian List, Nicholas L. Rodd, Geraint F. Lewis, and Ishaan Bhat, *Phys. Rev. Lett.* 125, [241102](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.241102), 2020 [[arXiv:2006.12504](https://arxiv.org/abs/2006.12504)].
 **→** Branch "**prl**"  

The "**main**" branch contains the most up-to-date version, which is under development.

*Author*: Florian List (University of Vienna).
*Contributions*: Nicholas L. Rodd (CERN).
   
For any queries, please contact me at florian dot list at univie dot ac dot at.

![NN_sketch](https://github.com/FloList/GCE_NN/blob/prl/pngs/NN_sketch.png)

**Disclaimer**:
The code in this repository borrows from several other GitHub repositories and other sources.
In particular, the neural network architecture is built upon *DeepSphere* ([Perraudin et al. 2019](http://arxiv.org/abs/1810.12186), [Defferrard et al. 2020](https://openreview.net/pdf?id=B1e3OlStPB)).

## Data
The *Fermi* dataset and the templates contained in this repository have been generated for the following papers:
* *Spurious point source signals in the galactic center excess*, Rebecca K. Leane and Tracy R. Slatyer, *Phys. Rev. Lett.* 125, [121105](https://link.aps.org/doi/10.1103/PhysRevLett.125.121105), 2020
[[arXiv:2002.12370](http://arxiv.org/abs/2002.12370)].
* *The enigmatic Galactic Center excess: Spurious point sources and signal mismodeling*, Rebecca K. Leane and Tracy R. Slatyer, *Phys. Rev. D* 102, [063019](https://link.aps.org/doi/10.1103/PhysRevD.102.063019), 2020 [[arXiv:2002.12371](http://arxiv.org/abs/2002.12371)].
 
The data is provided with the permission of the authors, and everybody using the data for a publication should cite these papers.



## Getting started


## Results for the *Fermi* data
* In the $\textit{Fermi}$ data, we find a faint Galactic Center Excess (GCE) described by a median source-count distribution (SCD) peaked at a flux of $\sim4 \times 10^{-11} \ \text{counts} \ \text{cm}^{-2} \ \text{s}^{-1}$ (corresponding to $\sim3 - 4$ expected counts per PS), which would require $N \sim \mathcal{O}(10^4)$ sources to explain the entire excess (median value $N = \text{29,300}$ across the sky). 
* Although faint, this SCD allows us to derive the constraint $\eta_P \leq 66\%$ for the Poissonian fraction of the GCE flux $\eta_P$ at 95% confidence, suggesting that a substantial amount of the GCE flux is due to PSs.

## Citations
```
@article{List_et_al_2021,
archivePrefix = {arXiv},
arxivId = {2107.09070},
author = {List, Florian and Rodd, Nicholas L. and Lewis, Geraint F.},
eprint = {2107.09070},
journal = {preprint},
title = {{Dim but not entirely dark: Extracting the Galactic Center Excess' source-count distribution with neural nets}},
url = {http://arxiv.org/abs/2107.09070},
year = {2021}
}
```

```
@article{List_et_al_2020,
archivePrefix = {arXiv},
arxivId = {2006.12504},
author = {List, Florian and Rodd, Nicholas L. and Lewis, Geraint F. and Bhat, Ishaan},
eprint = {2006.12504},
journal = {Physical Review Letters},
volume = {125},
number = {24},
pages = {241102},
title = {{Galactic Center Excess in a New Light: Disentangling the γ-Ray Sky with Bayesian Graph Convolutional Neural Networks}},
url = {https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.241102},
year = {2020}
}
```