
# GCE NN
This repository contains the `Tensorflow` implementation of the papers
* *Extracting the Galactic Center Excess' source-count distribution with neural nets*, Florian List, Nicholas L. Rodd, and Geraint F. Lewis, *Phys. Rev. D* 104, [123022](https://link.aps.org/doi/10.1103/PhysRevD.104.123022), 2021 [[arXiv:2107.09070](https://arxiv.org/abs/2107.09070)]. <br>
 **→** Branch "**prd**"  
* *Galactic Center Excess in a New Light: Disentangling the γ-Ray Sky with Bayesian Graph Convolutional Neural Networks*, Florian List, Nicholas L. Rodd, Geraint F. Lewis, and Ishaan Bhat, *Phys. Rev. Lett.* 125, [241102](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.241102), 2020 [[arXiv:2006.12504](https://arxiv.org/abs/2006.12504)]. <br>
 **→** Branch "**prl**"  

The "**main**" branch contains the most up-to-date version, which is under development.

*Author*: Florian List (University of Vienna). <br>
*Contributions*: Nicholas L. Rodd (CERN).
   
For any queries, please contact me at florian dot list at univie dot ac dot at.

<p align="center">
<img src="https://github.com/FloList/GCE_NN/blob/main/pngs/nn_sketch.png" height="200"/>
&ensp;
<img src="https://github.com/FloList/GCE_NN/blob/prl/pngs/NN_sketch.png" height="200"/>
<p/>

**Disclaimer**:
The code in this repository borrows from several other GitHub repositories and other publicly available sources.
In particular, the neural network architecture is built upon *DeepSphere* ([Perraudin et al. 2019](http://arxiv.org/abs/1810.12186), [Defferrard et al. 2020](https://openreview.net/pdf?id=B1e3OlStPB)).

## Data
The *Fermi* dataset and the templates contained in this repository have been generated for the following papers:
* *Spurious point source signals in the galactic center excess*, Rebecca K. Leane and Tracy R. Slatyer, *Phys. Rev. Lett.* 125, [121105](https://link.aps.org/doi/10.1103/PhysRevLett.125.121105), 2020
[[arXiv:2002.12370](http://arxiv.org/abs/2002.12370)].
* *The enigmatic Galactic Center excess: Spurious point sources and signal mismodeling*, Rebecca K. Leane and Tracy R. Slatyer, *Phys. Rev. D* 102, [063019](https://link.aps.org/doi/10.1103/PhysRevD.102.063019), 2020 [[arXiv:2002.12371](http://arxiv.org/abs/2002.12371)].
 
The data is made available with the permission of the authors, and everybody using the data for a publication should cite these papers.

(The data selection criteria can be found in arXiv:2107.09070.)

## Getting started
### Option 1: Try out the code in Google Colab 
The folder ```examples``` contains an example Jupyter notebook ```gce_nn_example_notebook_colab.ipynb``` that can be run 
in Google Colab. The Jupyter notebook will clone this Github repository into the (temporary) ```/content/``` folder
on Google Colab, so it is **not** needed to manually clone or download this repository to get started.
1. Open [Google Colab](https://colab.research.google.com/).
2. Click on "Github" in the top panel.
3. Enter ```https://github.com/FloList/GCE_NN``` and click on "Search".
4. Select ```examples/gce_nn_example_notebook_colab.ipynb```.
5. Select a GPU runtime (*Runtime -> Change runtime type -> GPU*).
6. Run the notebook. (*Note:* After this Github repository has been cloned by the Jupyter notebook and the packages have 
been installed, the Jupyter notebook kernel needs to be restarted as explained in the notebook.)

### Option 2: Clone the repository and run it locally on your computer
First, clone the repository via
````
git clone https://github.com/FloList/GCE_NN.git
````
*Warning*: This github repository is **quite large** (several hundred MBs) because it contains *Fermi* data and templates.

Then, ``cd`` into the directory
````
cd GCE_NN
````

We highly recommend using a new virtual environment for the GCE NN analysis in which all the required packages can be installed 
in isolation from the globally installed packages.

This can be done using ```venv```
````
python3.8 -m venv venv_gce_nn  # create the environment
source venv_gce_nn/bin/activate  # activate it
# deactivate   # to deactivate the environment
````
or if you are using ```pyenv```
```
pyenv virtualenv 3.8.0 venv_gce_nn  # create the environment
pyenv activate venv_gce_nn  # activate it
# pyenv deactivate   # to deactivate the environment
```
or if you are using ```conda```
````
conda create -n venv_gce_nn python=3.8.0 anaconda  # create the environment
conda activate venv_gce_nn  # activate it
# conda deactivate  # to deactivate the environment
````
Once you are inside the virtual environment, all the required dependencies can be installed from the ```requirements.txt```
file with
```
pip install -r requirements.txt
```
Afterwards, install the GCE NN package with
````
python setup.py install
````

Then, a good starting point is the Jupyter notebook ```gce_nn_example_notebook.ipynb``` in the ```examples``` folder, which performs a convolutional neural network-based analysis of γ-ray photon-count maps for a simple scenario. To consider a different scenario, generate a new parameter file in the ```parameter_files``` folder (for example by copying the file ```parameters.py``` and modifying the relevant settings). 

## Results for the *Fermi* data (from arXiv:2107.09070)
<p align="center">
<img src="https://github.com/FloList/GCE_NN/blob/main/pngs/fermi_plot.png" width="600"/>
<p/>

* In the *Fermi* data, we find a **faint** Galactic Center Excess (GCE) described by a median source-count distribution (SCD) peaked at a flux of ~ 4 × 10⁻¹¹ counts / cm² / s (corresponding to ~ 3 - 4 expected counts per PS), which would require N ~ O(10⁴) sources to explain the entire excess (median value N = 29,300 across the sky). 
* Although faint, this SCD allows us to derive the constraint ηₚ ≤ 66% for the Poissonian fraction of the GCE flux ηₚ at 95% confidence, suggesting that a substantial amount of the GCE flux is due to PSs.

## Citations
```
@article{List_et_al_2021,
archivePrefix = {arXiv},
arxivId = {2107.09070},
author = {List, Florian and Rodd, Nicholas L. and Lewis, Geraint F.},
eprint = {2107.09070},
journal = {Physical Review D},
volume = {104},
number = {12},
pages = {123022},
title = {{Extracting the Galactic Center Excess' source-count distribution with neural nets}},
url = {https://link.aps.org/doi/10.1103/PhysRevD.104.123022},
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
