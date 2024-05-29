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

import jax
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 35,
        "text.usetex": True,
    }
)

import src.optimization as optimization
import src.circuits as circuits
import src.gates as gates

seed = 35
print("seed:", seed)
key = jax.random.PRNGKey(seed)

cutoff = 20
gate_cutoff = 10
number_of_layers = 25
number_of_iterations = 1000
learning_rate = 1e-3
param_threshold = 1e-2

target_param = 0.01
target_hamiltonian = gates.cubic_phase_hamiltonian(cutoff)
target_gate = gates.get_unitary(target_param, target_hamiltonian)

number_of_runs = 10
save_every_n = 4
xvals = list(range(0, number_of_iterations, save_every_n))

loss_and_grad = jax.jit(
    jax.value_and_grad(circuits.evaluate_and_loss),
    static_argnames=["cutoff", "gate_cutoff"],
)

plt.figure(figsize=(10, 9))
for run in range(number_of_runs):
    key, weights = circuits.generate_random_weights(number_of_layers, key=key)
    circuit = circuits.get_circuit_definition(number_of_layers)

    optimizer = optimization.JaxAdamOptimizer(
        learning_rate, number_of_weights=len(weights)
    )

    print(f"{number_of_runs}/{run}. Begin optimization...")
    loss_progress = []
    for i in range(number_of_iterations):
        loss, grad = loss_and_grad(weights, circuit, target_gate, cutoff, gate_cutoff)
        weights = optimizer.update_weights(weights, grad)
        if i % save_every_n == 0:
            loss_progress.append(loss)
    print("Optimization ended.")

    plt.plot(xvals, loss_progress)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(
    f"plots/adam_starting_point_cp({target_param})_c{cutoff}_gc{gate_cutoff}_tau{param_threshold}_lr{learning_rate}_it{number_of_iterations}_l{number_of_layers}_s{seed}.png"
)
plt.clf()
