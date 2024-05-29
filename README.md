# PhotonicGateDecomposer
This repository provides a solution for decomposing continuous variable photonic quantum transformations into elementary photonic quantum gates using adaptive circuit compression. 

## Installation

The core implementation requires only the `jax` and `jaxlib` packages:
```
pip install jax
pip install jaxlib
```
or can be installed specifically using:
```
pip install -r core-requirements.txt
```
The repository also contains scripts for creating diagrams and benchmarks requiring additional packages that can be installed using:
```
pip install -r requirements.txt
```

## Contents
- `src/`:
  - `circuits.py`: Contains all the necessary code for creating, and evaluating CVQNN layers as well as their corresponding loss values.
  - `gate_stats.py`: This module serves as a convenient way to generate certain important statistics about a resulting circuit and return a Python dictionary that can be saved.
  - `gates.py`: It has functions related to calculating individual gates for the CVQNN layer and target unitaries, e.g. the cubic phase gate.
  - `optimization.py`: This module is responsible for performing gate synthesis with adaptive circuit compression and evaluating the final result by calculating average fidelity values.
  - `uniatry_generator.py`: This module provides a wider range of freedom in generating random unitaries from Hamiltonians that are polynomial in the ladder operators. One can provide the degree of the Hamiltonian and corresponding coefficients for each term.
- `gate_learning.py`: This script encapsulates all aspects of our work and performs gate synthesis with adaptive circuit compression. One can change the hyperparameters and target unitaries easily.
- `plotting_scripts/`: Various scripts for plotting different benchmarks investigating how certain hyperparameters and target gate choices alter the resulting circuit. Saved results can be found in the `results/` and `plots/` folders with corresponding names.

## Contributing

This repository contains the source code for my master's thesis "Decomposition of unitary matrices based on bosonic Hamiltonian operators into elementary photonic quantum gates". It is not perfect, and general improvements are welcome.

