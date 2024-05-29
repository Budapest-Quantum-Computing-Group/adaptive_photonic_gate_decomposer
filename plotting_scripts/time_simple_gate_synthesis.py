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

import jax
import jax.numpy as np

import src.gates as gates
import src.circuits as circuits
import src.optimization as optimization

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

##### Basic parameters ######
seed = 100
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 20
gate_cutoff = 10
number_of_layers = 25
number_of_iterations = 1000
learning_rate = 1e-3

target_param = 0.01
# NOTE: Increased cutoff due to cubic phase inaccuracy, and to better match SF results
target_hamiltonian = gates.cubic_phase_hamiltonian(cutoff + 20)[:cutoff, :cutoff]
target_gate = gates.get_unitary(target_param, target_hamiltonian)

##### Running the code #####
circuit = circuits.get_circuit_definition(number_of_layers)
key, weights = circuits.generate_random_weights(number_of_layers, key=key)

print("Compile gate synthesis...")
weights = optimization.jit_gate_synthesis(
    weights,
    circuit,
    target_gate,
    number_of_iterations,
    learning_rate,
    cutoff,
    gate_cutoff,
)
print("Compilation ended.")

print("Starting gate synthesis...")
start = time()
weights = optimization.jit_gate_synthesis(
    weights,
    circuit,
    target_gate,
    number_of_iterations,
    learning_rate,
    cutoff,
    gate_cutoff,
)
runtime = time() - start
learnt_unitary = circuits.evaluate_circuit(weights, circuit, cutoff)
avg_fid = optimization.avg_gate_fidelity(target_gate, learnt_unitary, gate_cutoff)

print(
    f"Gate synthesis finished {number_of_iterations} iterations in {runtime} seconds resulting in {avg_fid} fidelity."
)
