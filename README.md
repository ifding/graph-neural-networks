# Graph Neural Networks for Quantum Chemistry

Implementation and modification of Message Passing Neural Networks as explained in the article proposed by Gilmer et al. [1].

Requirements:
- python 3.5
- pytorch=0.1.12
- networkx=1.11
- tensorboard
- tensorboard_logger
- numpy
- joblib


## Setup

Using `conda create` command to create a `conda` environment.

    $ module add anaconda3/4.2.0
    $ conda create -n python-3.5 python=3.5
    $ source activate python-3.5


## Installation

    $ pip install numpy tensorboard tensorboard_logger joblib
    $ conda install -c rdkit rdkit 
    $ conda install networkx=1.11
    $ conda install pytorch=0.1.12 cuda75 -c soumith
    $ git clone https://github.com/ifding/graph-neural-networks.git
    $ cd graph-neural-networks


## Examples

### QM9

Download and convert QM9 data set:

    $ python3 download_data.py qm9 -p /scratch3/feid/mpnn-data/

Train and test MPNN (default) and MPNNv2 model with GPU (default) or not:

    $ python3 main.py --datasetPath /scratch3/feid/mpnn-data/qm9/dsgdb9nsd/

    $ python3 main.py --datasetPath /scratch3/feid/mpnn-data/qm9/dsgdb9nsd/ --no-cuda
    
    $ python3 main.py --datasetPath /scratch3/feid/mpnn-data/qm9/dsgdb9nsd/ --model MPNNv2
        
    $ python3 main.py --datasetPath /scratch3/feid/mpnn-data/qm9/dsgdb9nsd/ --no-cuda --model MPNNv2

    $ python3 main.py --datasetPath /scratch3/feid/mpnn-data/qm9/dsgdb9nsd/ --model MPNNv3
        
    $ python3 main.py --datasetPath /scratch3/feid/mpnn-data/qm9/dsgdb9nsd/ --no-cuda --model MPNNv3


## Bibliography

- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.
- [2] Schütt, Kristof T., et al. [Quantum-chemical insights from deep tensor neural networks](https://www.nature.com/articles/ncomms13890.pdf) Nature communications 8 (2017): 13890.
- [3] Duvenaud *et al.*, [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1606.09375), NIPS, 2015.
- [4] Li *et al.*, [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493), ICLR, 2016. 
- [5] Kipf *et al.*, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR, 2017
- [6] Defferrard *et al.*, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS, 2016. 
- [7] Kearnes *et al.*, [Molecular Graph Convolutions: Moving Beyond Fingerprints](https://arxiv.org/abs/1603.00856), JCAMD, 2016. 