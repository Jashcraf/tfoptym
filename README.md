# tfoptym: optimizing thin films in Python

tfoptym is an optimizing tool for homogeneous thin film stacks in Python 3.8+. It features a hot-swappable backend to support numpy functions from
- numpy
- cupy
- jax

to enable computations on GPU's, and using autodiff to compute analytical gradients.