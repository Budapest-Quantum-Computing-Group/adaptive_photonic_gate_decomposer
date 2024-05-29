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

seed = 11
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 20
gate_cutoff = 10
number_of_layers = 25
number_of_weight_iterations = 60
learning_rate = 1e-3

target_param = 0.01
target_hamiltonian = gates.cubic_phase_hamiltonian(cutoff)
target_gate = gates.get_unitary(target_param, target_hamiltonian)

##### Running the code #####

param_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
end_circuits = []
end_avg_fidelities = []

for i in range(len(param_thresholds)):
    circuit = circuits.get_circuit_definition(number_of_layers)
    weights, circuit = optimization.adaptive_circuit_compression(
        key,
        cutoff,
        gate_cutoff,
        number_of_layers,
        number_of_weight_iterations,
        learning_rate,
        target_gate,
        param_thresholds[i],
    )
    if len(weights) == 0:
        end_avg_fidelities.append(0)
    else:
        learnt_unitary = circuits.evaluate_circuit(weights, circuit, cutoff)
        avg_fid = optimization.avg_gate_fidelity(
            target_gate, learnt_unitary, gate_cutoff
        )
        end_avg_fidelities.append(avg_fid)
    end_circuits.append(circuit)

    stats = get_stats(
        seed,
        cutoff,
        gate_cutoff,
        number_of_layers,
        weights,
        circuit,
        target_param,
        target_gate,
        param_thresholds[i],
        learning_rate,
        number_of_weight_iterations,
    )

    np.save(
        f"results/increasing_tau/{i}cp({target_param})_s({seed})_c({cutoff})_gc({gate_cutoff})_lr({learning_rate})_tau({param_thresholds[i]})_iter({number_of_weight_iterations})_l({number_of_layers})_s{seed}",
        stats,
    )

##### Plotting results #####

circ_lens = [len(end_circuits[i]) for i in range(len(end_circuits))]
corr_end_avg_fidelities = end_avg_fidelities
# Given that the last itearations filters out everything
for i in range(len(param_thresholds)):
    if circ_lens[i] == 0:
        corr_end_avg_fidelities[i] = 0

plt.figure(figsize=(12, 9))
plt.plot(param_thresholds, circ_lens, marker="x", color="red")
plt.xticks([0.01, 0.05, 0.1, 0.15, 0.2])
plt.xlabel("Filter threshold")
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/increasing_tau/gate_num_cp({target_param})_c{cutoff}_gc{gate_cutoff}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
plt.plot(param_thresholds, corr_end_avg_fidelities, marker="x", color="red")
plt.xticks([0.01, 0.05, 0.1, 0.15, 0.2])
plt.xlabel("Filter threshold")
plt.ylabel("Fidelity")
plt.savefig(
    f"plots/increasing_tau/gate_fid_cp({target_param})_c{cutoff}_gc{gate_cutoff}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()

plt.figure(figsize=(12, 9))
labels = ["Total"]
end_circ_hist = [circuits.get_circuit_definition(number_of_layers)]
for i in range(len(param_thresholds)):
    labels.append(str(param_thresholds[i]))
    end_circ_hist.append(end_circuits[i])

plt.hist(end_circ_hist, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], label=labels)
plt.legend()
plt.xlabel("Gate types")
plt.xticks([0, 1, 2, 3], [r"$P$", r"$S$", r"$D$", r"$K$"])
plt.ylabel("Number of gates")
plt.savefig(
    f"plots/increasing_tau/gate_hist_cp({target_param})_c{cutoff}_gc{gate_cutoff}_lr{learning_rate}_it{number_of_weight_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()
