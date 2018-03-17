# Graph Neural Networks for Quantum Chemistry

Implementation and modification of Deep Tensor Neural Networks on graphs as explained in the article proposed by Schütt, Kristof T., et al. [1].

Requirements:
- python3
- ASE
- numpy
- tensorflow (>=1.0)


## Installation

    $ git clone https://github.com/ifding/graph-neural-networks.git
    $ cd graph-neural-networks
    $ pip install -r requirements.txt


## Examples

### QM9

Download and convert QM9 data set:

    $ python3 load_qm9.py <qm9destination>

Train QM9 energy (U0) prediction:

    $ python3 train_energy_force.py <qm9destination>/qm9.db ./modeldir ./split50k.npz 
        --ntrain 50000 --nval 10000 --fit_energy --atomref <qm9destination>/atomref.npz

## Bibliography

- [1] Schütt, Kristof T., et al. [Quantum-chemical insights from deep tensor neural networks](https://www.nature.com/articles/ncomms13890.pdf) Nature communications 8 (2017): 13890.
- [2] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.
- [3] Duvenaud *et al.*, [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1606.09375), NIPS, 2015.
- [4] Li *et al.*, [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493), ICLR, 2016. 
- [5] Kipf *et al.*, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR, 2017
- [6] Defferrard *et al.*, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS, 2016. 
- [7] Kearnes *et al.*, [Molecular Graph Convolutions: Moving Beyond Fingerprints](https://arxiv.org/abs/1603.00856), JCAMD, 2016. 