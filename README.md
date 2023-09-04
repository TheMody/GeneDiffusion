# Genetic Diffusion

The Repository for Generating synthetic gentics data with Diffusion Models.

## Abstract


## Install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3

for replication:
- `pip install wandb` for optional logging <3
- for easy replication use conda and environment.yml eg:
`$ conda env create -f environment.yml` and `$ conda activate sls3`


## Replicating Results
For replicating the Results run:

```
$ python train.py
```
The script expects the sigSNPs_pca.features.pkl to be in the data/sigSNPs_pca.features.pkl path. This data is not provided by this repository.


