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

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from time import time
from functools import partial

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation

import jax
import jax.numpy as np
import numpy

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 35,
        "text.usetex": True,
    }
)


import tensorflow as tf

tf.get_logger().setLevel("INFO")
tf.keras.backend.set_floatx("float64")
tf.autograph.set_verbosity(1, alsologtostdout=False)

# Force jax to use double
jax.config.update("jax_enable_x64", True)
# Force jax to use cpu
jax.config.update("jax_default_device", jax.devices("cpu")[0])
# Nicer print
float_formatter = "{:.8f}".format
np.set_printoptions(
    suppress=True,
    linewidth=200,
    threshold=sys.maxsize,
    formatter={"float_kind": float_formatter},
)


@jax.jit
def inner_double_factorial(i, a):
    return a.at[i].set((2 * i - 1) / (2 * i) * a[i - 1])


@partial(jax.jit, static_argnames=["cutoff"])
def double_factorial_array(cutoff):
    array = np.empty(shape=(cutoff + 1) // 2)
    array = array.at[0].set(1)

    return jax.lax.fori_loop(1, len(array), inner_double_factorial, array)


@partial(jax.jit, static_argnames=["cutoff"])
def circuit_loss(target_gate, learnt_gate, cutoff):
    L = learnt_gate[:, :cutoff]
    T = target_gate[:, :cutoff]

    return (cutoff - np.real(np.trace(T.conj().T @ L))) / (cutoff)


@partial(jax.jit, static_argnames=["cutoff"])
def squeezing_unitary(r, cutoff):

    sechr = 1.0 / np.cosh(r)
    A = np.tanh(r)
    sqrt_indices = np.sqrt(np.arange(cutoff))
    sechr_sqrt_indices = sechr * sqrt_indices
    A_conj_sqrt_indices = A * sqrt_indices

    first_row_nonzero = np.sqrt(double_factorial_array(cutoff)) * np.power(
        -A, np.arange(0, (cutoff + 1) // 2)
    )

    first_row = np.zeros(shape=cutoff, dtype=np.complex128)
    first_row = first_row.at[np.arange(0, cutoff, 2)].set(first_row_nonzero)
    roll_index = np.arange(-1, cutoff - 1)
    second_row = sechr_sqrt_indices * first_row[roll_index]

    matrix = np.empty((cutoff, cutoff), dtype=np.complex128)
    matrix = matrix.at[0].set(first_row)
    matrix = matrix.at[1].set(second_row)

    previous_previous = first_row
    previous = second_row

    def inner_squeezing(i, mppp):
        matrix = mppp[0]
        prev = mppp[1]
        prev_prev = mppp[2]

        current = (
            sechr_sqrt_indices * prev[roll_index]
            + A_conj_sqrt_indices[i - 1] * prev_prev
        ) / sqrt_indices[i]

        matrix = matrix.at[i].set(current)
        prev_prev = prev
        prev = current

        return (matrix, prev, prev_prev)

    matrix = jax.lax.fori_loop(
        2, cutoff, inner_squeezing, (matrix, previous, previous_previous)
    )[0]
    return np.sqrt(sechr) * np.transpose(np.stack(matrix))


def sf_sq_unitary(r, cutoff):
    prog = sf.Program(1)

    sf_params = []
    names = ["sq_r"]

    for i in range(1):
        sf_params_names = ["{}_{}".format(n, i) for n in names]
        sf_params.append(prog.params(*sf_params_names))

    sf_params = numpy.array(sf_params)

    @operation(1)
    def layer(i, q):
        Sgate(sf_params[i], 0.0) | q
        return q

    in_ket = numpy.zeros([cutoff, cutoff])
    numpy.fill_diagonal(in_ket, 1)

    with prog.context as q:
        Ket(in_ket) | q
        layer(0) | q[0]

    eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": cutoff})

    mapping = {
        p.name: w for p, w in zip(sf_params.flatten(), tf.convert_to_tensor([r]))
    }

    state = eng.run(prog, args=mapping).state
    ket = state.ket()

    return ket


if __name__ == "__main__":
    cutoffs = list(range(6, 501, 2))
    avg_over = 20
    warmup_amount = 5
    r = 0.1
    r_tf = tf.Variable(r)
    runtimes = []
    runtimes_sf = []
    for cutoff in cutoffs:
        print(cutoff)

        for _ in range(warmup_amount):
            U = squeezing_unitary(r, cutoff)

        start = time()
        for _ in range(avg_over):
            U = squeezing_unitary(r, cutoff)
        runtimes.append((time() - start) / avg_over)

        for _ in range(warmup_amount):
            sf_U = sf_sq_unitary(r_tf, cutoff)

        start = time()
        for _ in range(avg_over):
            sf_U = sf_sq_unitary(r_tf, cutoff)
        runtimes_sf.append((time() - start) / avg_over)

    ticks = [cutoffs[i] for i in range(0, len(cutoffs), ((len(cutoffs) // 10) + 1))]

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, runtimes, color="blue")
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel("Fidelity", rotation=270, labelpad=40)
    ax = plt.subplot(1, 1, 1)
    ax.yaxis.set_label_position("right")
    plt.ylabel("Runtimes (sec)")
    plt.savefig(
        f"plots/runtimes/squeezing/our_sq_comp_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmup_amount}.png"
    )
    plt.clf()

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, runtimes_sf, color="red")
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel("Runtime (sec)", rotation=270, labelpad=40)
    ax = plt.subplot(1, 1, 1)
    ax.yaxis.set_label_position("right")
    plt.savefig(
        f"plots/runtimes/squeezing/their_sq_comp_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmup_amount}.png"
    )
    plt.clf()

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, runtimes, color="blue", label="ours")
    plt.plot(cutoffs, runtimes_sf, color="red", label="SF")
    plt.legend()
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel("Runtimes (sec)")
    plt.savefig(
        f"plots/runtimes/squeezing/common_sq_comp_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmup_amount}.png"
    )
    plt.clf()
