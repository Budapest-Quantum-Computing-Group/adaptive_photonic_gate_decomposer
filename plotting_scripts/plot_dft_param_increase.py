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

seed = 55
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 18
gate_cutoff = 8
number_of_layers = 40
number_of_weight_iterations = 50
learning_rate = 1e-4
param_threshold = 5e-3

##### Running script #####

target_params = [0.01, 0.1, 0.2, 0.5, 0.75, 1.0]
end_avg_fidelities = []
end_circuits = []

for i in range(len(target_params)):
    print("Target param:", target_params[i])
    target_gate = gates.parameterized_dft(cutoff, target_params[i])

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
        target_params[i],
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
        f"results/dft_param_increase/{i}-dft({target_params[i]})_s({seed})_c({cutoff})_gc({gate_cutoff})_lr({learning_rate})_tau({param_threshold})_iter({number_of_weight_iterations})_l({number_of_layers})_s{seed}",
        stats,
    )

##### Plotting results ######

plt.figure(figsize=(12, 9))
plt.plot(
    target_params,
    [len(end_circuits[i]) for i in range(len(end_circuits))],
    marker="x",
    color="red",
)
plt.xlabel("Target gate parameter")
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/dft_param_increase/gate_num_dft({target_params})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
plt.plot(target_params, end_avg_fidelities, marker="x", color="red")
plt.xlabel("Target gate parameter")
plt.ylabel("Fidelity")
plt.savefig(
    f"plots/dft_param_increase/gate_fid_dft({target_params})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
labels = []
for param in target_params:
    labels.append(str(param))
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
    f"plots/dft_param_increase/gate_hist_dft({target_params})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_{seed}.png"
)
plt.clf()
