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

from time import time
from functools import partial

import jax
import jax.numpy as np

import src.circuits as circuits

# Force jax to use double
jax.config.update("jax_enable_x64", True)
# Force jax to use cpu
jax.config.update("jax_default_device", jax.devices("cpu")[0])


class JaxAdamOptimizer:
    def __init__(
        self,
        lr,
        number_of_weights,
        decay1=0.9,
        decay2=0.999,
    ):
        self.lr = lr
        self.decay1 = decay1
        self.decay2 = decay2
        self.first_moment = np.zeros(number_of_weights, dtype=float)
        self.second_moment = np.zeros(number_of_weights, dtype=float)
        self.step_count = 0

        self.epsilon = 1e-08

    def update_weights(self, weights: np.ndarray, gradient: np.ndarray):
        self.step_count += 1

        new_weights, self.first_moment, self.second_moment = adam_step(
            weights=weights,
            gradient=gradient,
            lr=self.lr,
            first_moment=self.first_moment,
            second_moment=self.second_moment,
            step_count=self.step_count,
            decay1=self.decay1,
            decay2=self.decay2,
            epsilon=self.epsilon,
        )

        return new_weights


@jax.jit
def adam_step(
    weights,
    gradient,
    lr,
    first_moment,
    second_moment,
    step_count,
    decay1=0.9,
    decay2=0.999,
    epsilon=1e-08,
):
    new_first_moment = decay1 * first_moment + (1 - decay1) * gradient
    new_second_moment = decay2 * second_moment + (1 - decay2) * (np.square(gradient))
    new_lr = lr * (np.sqrt(1 - decay2**step_count)) / (1 - decay1**step_count)
    new_weights = weights - new_lr * new_first_moment / (
        np.sqrt(new_second_moment) + epsilon
    )

    return np.array([new_weights, new_first_moment, new_second_moment])


@partial(jax.jit, static_argnames=["cutoff", "gate_cutoff"])
def jit_gate_synthesis(
    weights, circuit, target_gate, iternum, learning_rate, cutoff, gate_cutoff
):
    first_moment = np.zeros(len(weights), dtype=float)
    second_moment = np.zeros(len(weights), dtype=float)

    def inner_gate_synthesis(i, a):
        w = a[0]
        fm = a[1]
        sg = a[2]

        grad = circuits.evaluate_and_loss_grad(
            w, circuit, target_gate, cutoff, gate_cutoff
        )
        nw, nfm, nsg = adam_step(w, grad, learning_rate, fm, sg, i)

        return (nw, nfm, nsg)

    # NOTE: Starting from 1 is needed to avoid 0 division in adam
    return jax.lax.fori_loop(
        1, iternum, inner_gate_synthesis, (weights, first_moment, second_moment)
    )[0]


def gate_synthesis(
    weights,
    circuit,
    target_gate,
    iternum,
    learning_rate,
    cutoff,
    gate_cutoff,
    calc_loss=False,
    loss_threshold=1e-6,
    std_threshold=0.0015,
    last_k=1000,
):
    assert (
        len(weights) == len(circuit) and len(weights) > 0
    ), f"Possible size mismatch. len(weights): {len(weights)}, len(circuit): {len(circuit)}, or empty circuit"
    if not calc_loss:
        return jit_gate_synthesis(
            weights,
            circuit,
            target_gate,
            iternum,
            learning_rate,
            cutoff,
            gate_cutoff,
        )

    optimizer = JaxAdamOptimizer(learning_rate, number_of_weights=len(weights))
    loss_progress = np.full(last_k, np.nan)

    loss_and_grad = jax.jit(
        jax.value_and_grad(circuits.evaluate_and_loss),
        static_argnames=["cutoff", "gate_cutoff"],
    )

    for i in range(iternum):
        loss, grad = loss_and_grad(weights, circuit, target_gate, cutoff, gate_cutoff)
        loss_progress = loss_progress.at[i % last_k].set(loss)
        if i >= last_k - 1:
            std_dev = np.std(loss_progress)
            if std_dev <= std_threshold or loss <= loss_threshold:
                break
        weights = optimizer.update_weights(weights, grad)

    return weights


def filter_weights(weights, circuit, param_threshold):
    gate_num = len(circuit)

    # Phaseshift modulo
    weights = np.where(
        circuit == circuits.GateIdx.PS.value, weights % (2 * np.pi), weights
    )
    weights = np.where(
        (circuit == circuits.GateIdx.PS.value)
        & (np.abs(weights - np.pi * 2) < param_threshold),
        0,
        weights,
    )

    # Threshold filtering
    w_filter_mask = abs(weights) > param_threshold

    weights = weights[w_filter_mask]
    circuit = circuit[w_filter_mask]

    # Combining same neighbours
    list_circuit_def = [int(gate_idx) for gate_idx in circuit]
    list_weights = [float(weight) for weight in weights]

    i = 0
    while i < len(list_circuit_def) - 1:
        if list_circuit_def[i] == list_circuit_def[i + 1]:
            list_weights[i] += list_weights[i + 1]
            list_weights.pop(i + 1)
            list_circuit_def.pop(i + 1)
        else:
            i += 1

    circuit = np.array(list_circuit_def)
    weights = np.array(list_weights)

    return weights, circuit, gate_num - len(circuit)


def pick_starting_point(
    key,
    cutoff,
    gate_cutoff,
    number_of_layers,
    head_start_iterations,
    candidates,
    learning_rate,
    target_unitary,
):
    circuit = circuits.get_circuit_definition(number_of_layers)

    weights_list = []
    for _ in range(candidates):
        key, weights = circuits.generate_random_weights(number_of_layers, key=key)
        weights_list.append(weights)

    results = [
        jit_gate_synthesis(
            weights_list[i],
            circuit,
            target_unitary,
            head_start_iterations,
            learning_rate,
            cutoff,
            gate_cutoff,
        )
        for i in range(candidates)
    ]
    losses = [
        circuits.evaluate_and_loss(result, circuit, target_unitary, cutoff, gate_cutoff)
        for result in results
    ]

    best_idx = np.argmin(np.array(losses))

    return results[best_idx], losses[best_idx]


def adaptive_circuit_compression(
    key,
    cutoff,
    gate_cutoff,
    number_of_layers,
    number_of_weight_iterations,
    learning_rate,
    target_gate,
    param_thresholds,
    calc_loss=False,
    stop_threshold=0.0015,
    last_k=1000,
):
    print("Searching for a good starting point...")
    start = time()
    weights, loss = pick_starting_point(
        key, cutoff, gate_cutoff, number_of_layers, 500, 10, 0.001, target_gate
    )
    runtime = time() - start
    print(f"Search over, it took {runtime:.2f} seconds with loss {loss:.4f}.")
    circuit = circuits.get_circuit_definition(number_of_layers)

    number_of_filtered_weights = -1
    round_num = 0
    if not isinstance(param_thresholds, list):
        param_thresholds = [param_thresholds]

    while number_of_filtered_weights != 0 or len(weights) == 0:
        print("Starting gate synthesis...")

        weights = gate_synthesis(
            weights,
            circuit,
            target_gate,
            number_of_weight_iterations * len(weights),
            learning_rate,
            cutoff,
            gate_cutoff,
            calc_loss,
            stop_threshold,
            last_k,
        )

        learnt_unitary = circuits.evaluate_circuit(weights, circuit, cutoff)
        avg_fid = avg_gate_fidelity(target_gate, learnt_unitary, gate_cutoff)

        weights, circuit, number_of_filtered_weights = filter_weights(
            weights, circuit, param_thresholds[round_num]
        )
        round_num = min(round_num + 1, len(param_thresholds) - 1)

        print(
            f"Gate synthesis finished with average fidelity {avg_fid:.4f}, and filtered {number_of_filtered_weights} gates"
        )
        if len(weights) == 0:
            print("Circuit empty after filtering!")
            break

    print(
        f"{number_of_layers*6}/{len(weights)} gates remain, which is the {100*len(weights)/(number_of_layers*6):.2f}% of the initial gates"
    )
    return weights, circuit


@partial(jax.jit, static_argnames=["cutoff", "gate_cutoff"])
def process_fidelity(target_gate, learnt_gate, cutoff, gate_cutoff):
    e = np.identity(cutoff, dtype=np.complex128)
    psi1 = np.zeros(cutoff**2, dtype=np.complex128)
    psi2 = np.zeros(cutoff**2, dtype=np.complex128)

    @jax.jit
    def inner_process_fidelity(i, a):
        psi1 = a[0] + np.kron(e[i], learnt_gate @ e[i])
        psi2 = a[1] + np.kron(e[i], target_gate @ e[i])

        return (psi1, psi2)

    psi1, psi2 = jax.lax.fori_loop(0, gate_cutoff, inner_process_fidelity, (psi1, psi2))

    return np.abs(psi1.conj() @ psi2 / gate_cutoff) ** 2


@partial(jax.jit, static_argnames=["gate_cutoff"])
def avg_gate_fidelity(target_gate, learnt_gate, gate_cutoff):
    cutoff = learnt_gate.shape[0]
    proc_fid = process_fidelity(target_gate, learnt_gate, cutoff, gate_cutoff)

    return (proc_fid * gate_cutoff + 1) / (gate_cutoff + 1)
