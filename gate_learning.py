#
# Copyright 2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from time import time
import random

import jax
import jax.numpy as np

import src.gates as gates
from src.optimization import adaptive_circuit_compression
from src.gate_stats import get_stats, Stat
from src.unitary_generator import (
    generate_random_unitary,
    get_number_of_coefficients_of_degree,
)

# Force jax to use double
jax.config.update("jax_enable_x64", True)
# Nicer print
float_formatter = "{:.8f}".format
np.set_printoptions(
    suppress=True,
    linewidth=200,
    threshold=sys.maxsize,
    formatter={"float_kind": float_formatter},
)
# Force jax to use cpu
jax.config.update("jax_default_device", jax.devices("cpu")[0])

if __name__ == "__main__":
    seed = 123
    print("seed:", seed)
    key = jax.random.PRNGKey(seed)

    # Basic hyperparameters
    cutoff = 20
    gate_cutoff = 10
    number_of_layers = 25
    number_of_weight_iterations = 40
    learning_rate = 1e-3
    param_thresholds = [0.005, 0.015, 0.02]
    target_param = 0.01

    # Defining the target gate

    # General Hamiltonian
    # degree = 3
    # coeffs = np.ones(get_number_of_coefficients_of_degree(degree))
    # target_gate = generate_random_unitary(key, degree, cutoff, target_param, coeffs)

    # DFT matrix
    # target_gate = gates.parameterized_dft(cutoff, target_param)

    # Cupic phase
    target_hamiltonian = gates.cubic_phase_hamiltonian(cutoff + 20)[:cutoff, :cutoff]
    target_gate = gates.get_unitary(target_param, target_hamiltonian)

    print("Starting gate synthesis with adaptive circuit compression...")
    start = time()
    with jax.disable_jit(False):  # For debugging
        weights, circuit = adaptive_circuit_compression(
            key,
            cutoff,
            gate_cutoff,
            number_of_layers,
            number_of_weight_iterations,
            learning_rate,
            target_gate,
            param_thresholds,
        )
    runtime = (time() - start) / 60
    print(f"Full process ended in {runtime:.4f} minutes.")

    # Create summary
    stats = get_stats(
        seed,
        cutoff,
        gate_cutoff,
        number_of_layers,
        weights,
        circuit,
        target_param,
        target_gate,
        param_thresholds,
        learning_rate,
        number_of_weight_iterations,
    )

    print("Resulting fidelities:")
    print(f"Proc.:{stats[Stat.PFID]:.4f},Avg.:{stats[Stat.AVGFID]:.4f}")
    print(
        f"{number_of_layers*6}/{len(weights)} gates remain, which is the {100*len(weights)/(number_of_layers*6):.2f}% of the initial gates"
    )

    print("Maximum gate values:")
    print("Squeezing:", stats[Stat.MAXSQPARAM])
    print("Phaseshifter:", stats[Stat.MAXPSPARAM])
    print("Displacement:", stats[Stat.MAXDPPARAM])
    print("Kerr:", stats[Stat.MAXKRPARAM])

    np.save(
        f"results/cp({target_param})_s({seed})_c({cutoff})_gc({gate_cutoff})_lr({learning_rate})_tau({param_thresholds})_iter({number_of_weight_iterations})_l({number_of_layers})_s{seed}",
        stats,
    )
    # saved_dict = np.load("example.npy", allow_pickle=True).item()
