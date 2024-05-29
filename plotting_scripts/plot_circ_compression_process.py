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

import src.gates as gates
import src.circuits as circuits
import src.optimization as optimization
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

##### Basic parameters ######
seed = 30
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 20
gate_cutoff = 10
number_of_layers = 25
number_of_weight_iterations = 60
learning_rate = 1e-3
param_thresholds = 0.01

target_param = 0.01
# NOTE: Increased cutoff due to cubic phase inaccuracy, and to better match SF results
target_hamiltonian = gates.cubic_phase_hamiltonian(cutoff + 20)[:cutoff, :cutoff]
target_gate = gates.get_unitary(target_param, target_hamiltonian)

##### Running the code #####
circuit = circuits.get_circuit_definition(number_of_layers)
key, weights = circuits.generate_random_weights(number_of_layers, key=key)

gate_nums = []
avg_fids = []
number_of_filtered_weights = -1
while number_of_filtered_weights != 0:
    print("Starting gate synthesis...")

    weights = optimization.jit_gate_synthesis(
        weights,
        circuit,
        target_gate,
        number_of_weight_iterations * number_of_layers,
        learning_rate,
        cutoff,
        gate_cutoff,
    )

    learnt_unitary = circuits.evaluate_circuit(weights, circuit, cutoff)
    avg_fid = optimization.avg_gate_fidelity(target_gate, learnt_unitary, gate_cutoff)

    avg_fids.append(avg_fid)
    gate_nums.append(len(circuit))
    weights, circuit, number_of_filtered_weights = optimization.filter_weights(
        weights, circuit, param_thresholds
    )

    print(
        f"Gate synthesis finished with fidelity {avg_fid:.4f}, and filtered {number_of_filtered_weights} gates"
    )

print(
    f"{number_of_layers*6}/{len(weights)} gates remain, which is the {100*len(weights)/(number_of_layers*6):.2f}% of the initial gates"
)

##### Plotting results #####
plt.figure(figsize=(12, 9))
rounds = list(range(1, len(gate_nums) + 1))
plt.plot(rounds, gate_nums, marker="x", color="red")
plt.xticks(rounds)
plt.xlabel("Optimization round")
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/circ_compression_process/gate_num_cp({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_thresholds}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
plt.plot(rounds, avg_fids, marker="x", color="red")
plt.xticks(rounds)
plt.xlabel("Optimization round")
plt.ylabel("Fidelity", rotation=270, labelpad=40)
ax = plt.subplot(1, 1, 1)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
ax.yaxis.set_label_position("right")
plt.savefig(
    f"plots/circ_compression_process/gate_fid_cp({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_thresholds}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
labels = ["Total", "Compressed"]
plt.hist(
    [circuits.get_circuit_definition(number_of_layers), circuit],
    bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
    label=labels,
)
plt.legend()
plt.xlabel("Gate types")
plt.xticks([0, 1, 2, 3], [r"$P$", r"$S$", r"$D$", r"$K$"])
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/circ_compression_process/gate_hist_cp({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_thresholds}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

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

print("Maximum gate values:")
print("Squeezing:", stats[Stat.MAXSQPARAM])
print("Phaseshifter:", stats[Stat.MAXPSPARAM])
print("Displacement:", stats[Stat.MAXDPPARAM])
print("Kerr:", stats[Stat.MAXKRPARAM])

np.save(
    f"results/circ_compression_process/cp({target_param})_s({seed})_c({cutoff})_gc({gate_cutoff})_lr({learning_rate})_tau({param_thresholds})_iter({number_of_weight_iterations})_l({number_of_layers})_s{seed}",
    stats,
)
