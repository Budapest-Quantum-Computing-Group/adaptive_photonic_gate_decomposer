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

from functools import partial

from scipy.linalg import dft
import jax
import jax.numpy as np
import jax.scipy as sp


@partial(jax.jit, static_argnames=["cutoff"])
def _double_factorial_array(cutoff):
    array = np.empty(shape=(cutoff + 1) // 2)
    array = array.at[0].set(1)

    def inner_double_factorial(i, a):
        return a.at[i].set((2 * i - 1) / (2 * i) * a[i - 1])

    return jax.lax.fori_loop(1, len(array), inner_double_factorial, array)


def _DFT(cutoff):
    return np.array((dft(cutoff) / np.sqrt(cutoff)))


def parameterized_dft(cutoff, param):
    # NOTE: For certain cutoff values it might not be unitary
    D, U = np.linalg.eig(_DFT(cutoff))
    A = U @ np.diag(np.log(D)) @ np.linalg.inv(U)

    unitary = jax.scipy.linalg.expm(param * A)

    assert np.allclose(unitary.conj().T @ unitary, np.identity(cutoff))

    return unitary


def get_creation_operator(cutoff):
    return np.diag(np.sqrt(np.arange(1, cutoff, dtype=np.complex128)), -1)


def cubic_phase_hamiltonian(cutoff):
    a_dagger = get_creation_operator(cutoff)
    x = a_dagger + a_dagger.T
    return x @ x @ x


@jax.jit
def get_unitary(param, hamiltonian):
    unitary = sp.linalg.expm(1j * param * hamiltonian)
    return unitary


@partial(jax.jit, static_argnames=["cutoff"])
def phaseshift_unitary(phi, cutoff):
    return np.diag(np.exp(1j * phi * np.arange(cutoff)))


@partial(jax.jit, static_argnames=["cutoff"])
def kerr_unitary(kappa, cutoff):
    return np.diag(np.exp(1j * kappa * (np.square(np.arange(cutoff)))))


@partial(jax.jit, static_argnames=["cutoff"])
def displacement_unitary(r, cutoff):
    cutoff_range = np.arange(cutoff)
    sqrt_indices = np.sqrt(cutoff_range)
    denominator = 1 / np.sqrt(jax.scipy.special.factorial(cutoff_range))

    previous_element = np.power(r, cutoff_range) * denominator
    matrix = np.empty((cutoff, cutoff), dtype=np.complex128)
    matrix = matrix.at[0].set(previous_element)
    roll_index = np.arange(-1, cutoff - 1)

    def inner_displacement(i, mp):
        matrix = mp[0]
        prev = mp[1]
        prev = sqrt_indices * prev[roll_index] - r * prev
        return (matrix.at[i].set(prev), prev)

    matrix = jax.lax.fori_loop(
        1, cutoff, inner_displacement, (matrix, previous_element)
    )[0]

    return (
        np.exp(-0.5 * r**2) * np.transpose(np.stack(matrix)) * denominator
    ).astype(np.complex128)


@partial(jax.jit, static_argnames=["cutoff"])
def squeezing_unitary(r, cutoff):

    sechr = 1.0 / np.cosh(r)
    A = np.tanh(r)
    sqrt_indices = np.sqrt(np.arange(cutoff))
    sechr_sqrt_indices = sechr * sqrt_indices
    A_conj_sqrt_indices = A * sqrt_indices

    first_row_nonzero = np.sqrt(_double_factorial_array(cutoff)) * np.power(
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
