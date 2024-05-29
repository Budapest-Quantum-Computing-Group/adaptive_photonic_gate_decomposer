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

import random
import sys

from time import time
from functools import partial

import piquasso as pq
import numpy
import jax
from jax import jit, value_and_grad
import jax.numpy as np
import optax

from scipy.special import comb

import src.optimization as optimization
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20})

from src.circuits import circuit_loss

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
# jax.config.update('jax_default_device', jax.devices('cpu')[0])


decorator = partial(jit, static_argnames=["cutoff"])


@decorator
def evaluate_pq_circuit(weights, cutoff):
    d = pq.cvqnn.get_number_of_modes(weights.shape[1])
    calculator = pq.JaxCalculator()
    config = pq.Config(cutoff=cutoff, normalize=False)

    simulator = pq.PureFockSimulator(
        d,
        config=config,
        calculator=calculator,
    )

    initial_state = pq.BatchPureFockState(d=d, calculator=calculator, config=config)
    initial_state.state_vector = numpy.identity(
        target_gate.shape[0], dtype=config.complex_dtype
    )

    cvqnn_layers = pq.cvqnn.create_layers(weights)
    program = pq.Program(instructions=cvqnn_layers.instructions)

    final_state = simulator.execute(initial_state=initial_state, program=program).state

    return final_state.state_vector


def get_target_gate(param, cutoff, d):
    config = pq.Config(cutoff=cutoff, normalize=False)
    simulator = pq.PureFockSimulator(d=d, config=config)

    state = pq.BatchPureFockState(d=d, calculator=pq.NumpyCalculator(), config=config)

    glob_cutoff = int(comb(d + cutoff - 1, cutoff - 1))
    state.state_vector = numpy.identity(glob_cutoff, dtype=config.complex_dtype)

    result = simulator.execute_instructions(
        initial_state=state, instructions=[pq.CrossKerr(param)]
    )

    return result.state.state_vector


@partial(jit, static_argnames=["cutoff", "global_gate_cutoff"])
@value_and_grad
def pq_loss(weights, target_gate, cutoff, global_gate_cutoff):
    learnt_gate = evaluate_pq_circuit(weights, cutoff)
    return circuit_loss(target_gate, learnt_gate, global_gate_cutoff)


if __name__ == "__main__":
    seed = 123
    print("seed:", seed)
    numpy.random.seed(seed)  # For Piquasso random weight generation

    d = 2
    cutoff = 12
    gate_cutoff = 6
    global_gate_cutoff = int(comb(d + gate_cutoff - 1, gate_cutoff - 1))
    layer_count = 5

    target_param = 0.1
    target_gate = get_target_gate(param=target_param, cutoff=cutoff, d=d)
    print(
        f"Target shape: {target_gate.shape}, global gate cutoff: {global_gate_cutoff}"
    )

    learning_rate = 0.001
    iterations = 4000
    weights = pq.cvqnn.generate_random_cvqnn_weights(layer_count, d)

    print("Starting compilation...")
    start = time()
    learnt_gate = evaluate_pq_circuit(weights, cutoff)
    grads = pq_loss(weights, target_gate, cutoff, global_gate_cutoff)
    compile_time_mins = (time() - start) / 60
    print(f"Compilation ended in: {compile_time_mins} minutes")

    plt.figure(figsize=(10, 7))
    cut_first_n = 250
    xvals = list(range(cut_first_n, iterations))
    number_of_runs = 5
    for run in range(number_of_runs):
        weights = pq.cvqnn.generate_random_cvqnn_weights(layer_count, d)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(weights)
        loss_progress = []
        start = time()
        print("Starting optimization...")
        for i in range(iterations):
            learnt_gate = evaluate_pq_circuit(weights, cutoff)
            loss, grads = pq_loss(weights, target_gate, cutoff, global_gate_cutoff)

            updates, opt_state = optimizer.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)

            loss_progress.append(loss)
            if i % (iterations // 20) == 0:
                print(f"{i}. loss: {loss}")

        print(
            f"{number_of_runs}/{run} Cross-Kerr synthesis ended in {time() - start} seconds for {iterations} iterations with {layer_count} number of layers for ({cutoff}, {gate_cutoff}) initial cutoff and gate cutoff values."
        )
        print(
            f"Avg. fidelity: {optimization.avg_gate_fidelity(target_gate, evaluate_pq_circuit(weights, cutoff), global_gate_cutoff)}"
        )

        plt.plot(xvals, loss_progress[cut_first_n:])

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(
    f"plots/circ_compression_process/pq_cross-kerr({target_param})_c{cutoff}_gc{gate_cutoff}_lr{learning_rate}_it{iterations}_l{layer_count}_s{seed}.png"
)
plt.clf()
