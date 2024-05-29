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
from enum import Enum
from functools import partial

import jax
import jax.numpy as np

import src.gates as gates


class GateIdx(Enum):
    PS = 0
    SQ = 1
    DP = 2
    KR = 3


def get_circuit_definition(layer_num):
    circuit_definition = np.array(
        [
            GateIdx.PS.value,
            GateIdx.SQ.value,
            GateIdx.PS.value,
            GateIdx.DP.value,
            GateIdx.PS.value,
            GateIdx.KR.value,
        ],
        dtype=int,
    )

    return np.tile(circuit_definition, layer_num)


def get_circuit_functions(cutoff):
    @jax.jit
    def psu(w):
        return gates.phaseshift_unitary(w, cutoff)

    @jax.jit
    def squ(w):
        return gates.squeezing_unitary(w, cutoff)

    @jax.jit
    def dpu(w):
        return gates.displacement_unitary(w, cutoff)

    @jax.jit
    def kru(w):
        return gates.kerr_unitary(w, cutoff)

    return psu, squ, dpu, kru


def generate_random_weights(number_of_layers, key=None, seed=None):
    if key is None:
        if seed is None:
            seed = random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)

    passive_sd = 0.1
    active_sd = 0.01

    key, subkey = jax.random.split(key)
    sq_r = active_sd * jax.random.normal(
        subkey, shape=(number_of_layers,), dtype=np.float64
    )
    key, subkey = jax.random.split(key)
    d_r = active_sd * jax.random.normal(
        subkey, shape=(number_of_layers,), dtype=np.float64
    )
    key, subkey = jax.random.split(key)
    r1 = passive_sd * jax.random.normal(
        subkey, shape=(number_of_layers,), dtype=np.float64
    )
    key, subkey = jax.random.split(key)
    r2 = passive_sd * jax.random.normal(
        subkey, shape=(number_of_layers,), dtype=np.float64
    )
    key, subkey = jax.random.split(key)
    r3 = passive_sd * jax.random.normal(
        subkey, shape=(number_of_layers,), dtype=np.float64
    )
    key, subkey = jax.random.split(key)
    kappa = active_sd * jax.random.normal(
        subkey, shape=(number_of_layers,), dtype=np.float64
    )

    weights = (np.array([r1, sq_r, r2, d_r, r3, kappa])).T
    weights = np.reshape(weights, weights.shape[0] * weights.shape[1])
    return subkey, weights


@partial(jax.jit, static_argnames=["cutoff"])
def evaluate_circuit(weights, circuit, cutoff):
    funcs = get_circuit_functions(cutoff)

    def inner_evaluate_circuit(i, a):
        gate = jax.lax.switch(circuit[i], funcs, *(weights[i],))
        return np.dot(gate, a)

    return jax.lax.fori_loop(
        0,
        weights.shape[0],
        inner_evaluate_circuit,
        np.identity(cutoff, dtype=np.complex128),
    )


@partial(jax.jit, static_argnames=["gate_cutoff"])
def circuit_loss(target_gate, learnt_gate, gate_cutoff):
    L = learnt_gate[:, :gate_cutoff]
    T = target_gate[:, :gate_cutoff]

    return (gate_cutoff - np.real(np.trace(T.conj().T @ L))) / (2 * gate_cutoff)


@partial(jax.jit, static_argnames=["cutoff", "gate_cutoff"])
def evaluate_and_loss(weights, circuit, target_gate, cutoff, gate_cutoff):
    learnt_gate = evaluate_circuit(weights, circuit, cutoff)
    return circuit_loss(target_gate, learnt_gate, gate_cutoff)


@partial(jax.jit, static_argnames=["cutoff", "gate_cutoff"])
@jax.grad
def evaluate_and_loss_grad(weights, circuit, target_gate, cutoff, gate_cutoff):
    learnt_gate = evaluate_circuit(weights, circuit, cutoff)
    return circuit_loss(target_gate, learnt_gate, gate_cutoff)
