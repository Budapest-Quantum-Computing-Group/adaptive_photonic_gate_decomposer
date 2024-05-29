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

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 35,
        "text.usetex": True,
    }
)

import src.circuits as circuits
import src.optimization as optimization
from src.unitary_generator import (
    generate_random_unitary,
    get_number_of_coefficients_of_degree,
)
from src.gate_stats import get_stats, Stat

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

##### Basic parameters #####

seed = 43
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 20
gate_cutoff = 10
number_of_layers = 30
number_of_weight_iterations = 40
learning_rate = 1e-3
param_threshold = 1e-2
target_param = 0.01

##### Running script #####

degrees = np.arange(1, 6)
end_circuits = []
end_avg_fidelities = []

for i in range(len(degrees)):
    coeffs = np.ones(get_number_of_coefficients_of_degree(degrees[i]))
    print("Degree:", degrees[i])
    target_gate = generate_random_unitary(key, degrees[i], cutoff, target_param, coeffs)

    weights, circuit = optimization.adaptive_circuit_compression(
        key,
        cutoff,
        gate_cutoff,
        number_of_layers,
        number_of_weight_iterations,
        learning_rate,
        target_gate,
        param_threshold,
    )

    learnt_unitary = circuits.evaluate_circuit(weights, circuit, cutoff)
    avg_fid = optimization.avg_gate_fidelity(target_gate, learnt_unitary, gate_cutoff)

    end_circuits.append(circuit)
    end_avg_fidelities.append(avg_fid)

    stats = get_stats(
        seed,
        cutoff,
        gate_cutoff,
        number_of_layers,
        weights,
        circuit,
        target_param,
        target_gate,
        param_threshold,
        learning_rate,
        number_of_weight_iterations,
    )

    print("Maximum gate values:")
    print("Squeezing:", stats[Stat.MAXSQPARAM])
    print("Phaseshifter:", stats[Stat.MAXPSPARAM])
    print("Displacement:", stats[Stat.MAXDPPARAM])
    print("Kerr:", stats[Stat.MAXKRPARAM])

    np.save(
        f"results/increasing_hamiltonian_order/{i}-ham({target_param})_s({seed})_c({cutoff})_gc({gate_cutoff})_lr({learning_rate})_tau({param_threshold})_iter({number_of_weight_iterations})_l({number_of_layers})_s{seed}",
        stats,
    )

##### Plotting results #####

circ_lens = [len(end_circuits[i]) for i in range(len(end_circuits))]
plt.figure(figsize=(12, 9))
plt.plot(degrees, circ_lens, marker="x", color="red")
plt.xticks(degrees)
plt.xlabel("Degree")
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/increasing_hamiltonian_order/gate_num_ham({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
plt.plot(degrees, end_avg_fidelities, marker="x", color="red")
plt.xticks(degrees)
plt.xlabel("Degree")
plt.ylabel("Fidelity")
plt.savefig(
    f"plots/increasing_hamiltonian_order/gate_fid_ham({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
labels = []
for degree in degrees:
    labels.append(str(degree))
labels.reverse()
labels.insert(0, "Total")
end_circuits.reverse()
end_circuits.insert(0, circuits.get_circuit_definition(number_of_layers))

plt.hist(end_circuits, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], label=labels)
plt.legend()
plt.xlabel("Gate types")
plt.xticks([0, 1, 2, 3], [r"$P$", r"$S$", r"$D$", r"$K$"])
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/increasing_hamiltonian_order/gate_hist_ham({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()
