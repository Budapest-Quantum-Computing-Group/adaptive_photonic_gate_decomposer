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
from src.gate_stats import get_stats, Stat
from src.optimization import (
    pick_starting_point,
    filter_weights,
    avg_gate_fidelity,
    JaxAdamOptimizer,
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

##### Basic parameters ######

seed = 70
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 20
gate_cutoff = 10
number_of_layers = 25
number_of_weight_iterations = 30
learning_rate = 1e-4
param_threshold = 15e-3

target_param = 0.05
target_layers = 1
target_circ_def = circuits.get_circuit_definition(target_layers)
target_weights = np.array([target_param] * (target_layers * 6))
target_gate = circuits.evaluate_circuit(target_weights, target_circ_def, cutoff)

##### Running the code #####

print("Searching for a good starting point...")
weights, loss = pick_starting_point(
    key, cutoff, gate_cutoff, number_of_layers, 500, 10, 0.001, target_gate
)
print(f"Search over with loss {loss:.8f}.")
circuit = circuits.get_circuit_definition(number_of_layers)

gate_nums = []
avg_fids = []
number_of_filtered_weights = -1
while number_of_filtered_weights != 0:
    print("Starting gate synthesis...")

    optimizer = JaxAdamOptimizer(learning_rate, number_of_weights=len(weights))

    number_of_iterations = number_of_weight_iterations * len(weights)
    for i in range(number_of_iterations):
        grad = circuits.evaluate_and_loss_grad(
            weights, circuit, target_gate, cutoff, gate_cutoff
        )
        weights = optimizer.update_weights(weights, grad)

    learnt_unitary = circuits.evaluate_circuit(weights, circuit, cutoff)
    avg_fid = avg_gate_fidelity(target_gate, learnt_unitary, gate_cutoff)
    avg_fids.append(avg_fid)
    gate_nums.append(len(circuit))
    weights, circuit, number_of_filtered_weights = filter_weights(
        weights, circuit, param_threshold
    )
    print(
        f"Gate synthesis finished with average fidelity {avg_fid:.4f}, and filtered {number_of_filtered_weights} gates"
    )

print(
    f"{number_of_layers*6}/{len(weights)} gates remain, which is the {100*len(weights)/(number_of_layers*6):.2f}% of the initial gates"
)
print(f"Target circuit has {len(target_circ_def)} gates.")

##### Plotting results #####

plt.figure(figsize=(12, 9))
rounds = list(range(1, len(gate_nums) + 1))
plt.plot(rounds, gate_nums, marker="x", color="red")
plt.axhline(
    y=len(target_circ_def), color="g", linestyle="--", label="Number of gates in target"
)
plt.xticks(rounds)
plt.xlabel("Optimization round")
plt.ylabel("Number of gates")
plt.legend()
plt.savefig(
    f"plots/learn_gate_set/gate_num_gateset_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
plt.plot(rounds, avg_fids, marker="x", color="red")
plt.xticks(rounds)
plt.xlabel("Optimization round")
plt.ylabel("Fidelity", rotation=270, labelpad=40)
ax = plt.subplot(1, 1, 1)
ax.yaxis.set_label_position("right")
plt.savefig(
    f"plots/learn_gate_set/gate_fid_gateset_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
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
    f"plots/learn_gate_set/gate_hist_gateset_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
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
    f"results/learn_gate_set/gate_set({target_param})_s({seed})_c({cutoff})_gc({gate_cutoff})_lr({learning_rate})_tau({param_threshold})_iter({number_of_weight_iterations})_l({number_of_layers})_s{seed}",
    stats,
)
