# GCE NN
This is the Tensorflow implementation of the paper "The GCE in a New Light: Disentangling the γ-ray Sky with Bayesian Graph Convolutional Neural Networks" by Florian List, Nicholas L. Rodd, Geraint F. Lewis, Ishaan Bhat ([arXiv:2006.12504](https://arxiv.org/abs/2006.12504), accepted by *Phys. Rev. Lett.*).

![NN_sketch](https://github.com/FloList/GCE_NN/blob/main/pngs/NN_sketch.png)

*Author*: Florian List (Sydney Institute for Astronomy, School of Physics, A28, The University of Sydney, NSW 2006, Australia).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
   
For any queries, please contact me at florian dot list at sydney dot edu dot au.

## Requirements
The code is written in ```Python 3``` and has been tested with the following packages: ```cloudpickle 1.3.0``` or ```dill 0.3.2``` (for pickling lambda-functions), ```corner 2.0.1```, ```healpy 1.13.0```, ```matplotlib 3.1.1```, [```NPTFit 0.2```](https://github.com/bsafdi/NPTFit), ```numba 0.48.0```, ```numba 0.48.0```, ```psutil 5.7.0```, ```pymultinest 2.9```, ```ray 0.8.2```, ```scikit-learn 0.22.2```, ```scipy 1.4.1```, ```seaborn 0.10.0```, ```tensorboard 2.1.1```, ```tensorflow 2.1.0 ``` (currently in compatibility mode for TF 1), ```tensorflow-probability 0.9.0```, ```timeout-decorator 0.4.1```.

For generating point-source (PS) template maps, the tool [```NPTFit-Sim```](https://github.com/nickrodd/NPTFit-Sim) needs to be used.

This implementation borrows code from:
* https://github.com/deepsphere/DeepSphere
* https://github.com/mdeff/cnn_graph/
* https://github.com/deepsphere/paper-deepsphere-iclr2020/blob/51260d9169b9bff2f0d71d567c99909a17efd5e9/figures/kernel_widths.py
* https://github.com/yaringal/ConcreteDropout
* https://github.com/bsafdi/NPTFit/tree/master/examples

In particular, the neural network architecture is built on *DeepSphere* ([Perraudin et al. 2019](http://arxiv.org/abs/1810.12186), [Defferrard et al. 2020](https://openreview.net/pdf?id=B1e3OlStPB)).

## Training the neural network
A good starting point is the Jupyter notebook ```Train_NN_Poisson_only.ipynb``` that shows how the neural network can be trained for the case of photon maps being composed of Poissonian templates only. For this case, no PS maps need to be generated, and the training and testing photon count maps are randomly generated on-the-fly with randomly drawn normalisations as specified by the priors. The hyperparameters need to be specified in the file ```parameters_CNN.py```. 

For including PSs, the script ```generate_data_per_model.py``` in the folder ```Scripts``` needs to be run in order to generate PS maps. Once the PS map generation is done, there are two options:
 1. Combine the PS maps with Poissonian template maps using the script ```combine_data_from_models.py```. Then, train the NN with NN type `CNN_pre` (hyperparameters in ```CNN_parameters_pre_gen.py```).
 2. Combine the PS maps with Poissonian template maps that are generated on-the-fly during the NN training. This can be done by setting the NN type to `CNN` (hyperparameters in ```CNN_parameters.py```).

An experimental U-Net implementation for the pixelwise regression is also included (the hyperparameters must be specified in the file ```parameters_UNet.py```), but has not been tested properly. Also, many options (in particular for the estimation of uncertainties) have not been fully implemented, yet, and not all combinations of hyperparameters / settings are supported. This github repo contains the current state of our neural network implementation and does not represent an official code release.

## Results for the *Fermi* map:
![Fermi_results](https://github.com/FloList/GCE_NN/blob/main/pngs/Predictions_for_Fermi.png)
An interactive version of this plot is available [here](https://zenodo.org/record/4044689/).



