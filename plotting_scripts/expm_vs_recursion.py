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

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 35,
        "text.usetex": True,
    }
)

from src.gates import get_creation_operator, get_unitary, squeezing_unitary

# Force jax to use double
jax.config.update("jax_enable_x64", True)
# Force jax to use cpu
jax.config.update("jax_default_device", jax.devices("cpu")[0])


def squeezing_hamiltonian(cutoff):
    a_dagger = get_creation_operator(cutoff)
    return -1j * (a_dagger @ a_dagger - a_dagger.T @ a_dagger.T) / 2


if __name__ == "__main__":
    cutoffs = list(range(6, 301))
    param = 0.1

    exp_runtimes = []
    rec_runtimes = []
    differences = []

    avg_over = 300
    warmups = 20
    for cutoff in cutoffs:
        print("Cutoff:", cutoff)
        sq_ham = squeezing_hamiltonian(cutoff)

        for _ in range(warmups):
            U = get_unitary(param, sq_ham)

        start = time()
        for _ in range(avg_over):
            exp_U = get_unitary(param, sq_ham)
        exp_runtimes.append(1000 * (time() - start) / avg_over)

        for _ in range(warmups):
            U = squeezing_unitary(param, cutoff)

        start = time()
        for _ in range(avg_over):
            rec_U = squeezing_unitary(param, cutoff)
        rec_runtimes.append(1000 * (time() - start) / avg_over)

        differences.append(np.sum(np.abs(rec_U - exp_U)) / (cutoff * cutoff))

    ticks = [cutoffs[i] for i in range(0, len(cutoffs), ((len(cutoffs) // 10) + 1))]

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, rec_runtimes, color="blue")
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel("Fidelity", rotation=270, labelpad=40)
    ax = plt.subplot(1, 1, 1)
    ax.yaxis.set_label_position("right")
    plt.ylabel("Runtimes (ms)")
    plt.savefig(
        f"plots/runtimes/exp_vs_rec/rec_sq_comp_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmups}.png"
    )
    plt.clf()

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, exp_runtimes, color="red")
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel("Runtimes (ms)", rotation=270, labelpad=40)
    ax = plt.subplot(1, 1, 1)
    ax.yaxis.set_label_position("right")
    plt.savefig(
        f"plots/runtimes/exp_vs_rec/exp_sq_comp_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmups}.png"
    )
    plt.clf()

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, exp_runtimes, color="blue", label="exp")
    plt.plot(cutoffs, rec_runtimes, color="red", label="rec")
    plt.legend()
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel("Runtimes (ms)")
    plt.savefig(
        f"plots/runtimes/exp_vs_rec/common_sq_comp_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmups}.png"
    )
    plt.clf()

    plt.figure(figsize=(12, 9))
    plt.plot(cutoffs, differences)
    plt.xticks(ticks)
    plt.xlabel("Cutoff")
    plt.ylabel(r"$\Delta_{S,S'}$")
    plt.savefig(
        f"plots/runtimes/exp_vs_rec/element_diff_c{(cutoffs[0], cutoffs[-1])}_avg{avg_over}_wup{warmups}.png"
    )
    plt.clf()
