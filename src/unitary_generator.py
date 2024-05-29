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

import itertools
import random

import jax
import jax.numpy as np

from src.gates import get_creation_operator, get_unitary

# Force jax to use double
jax.config.update("jax_enable_x64", True)


def get_number_of_coefficients_of_degree(degree):
    return (degree + 1) * (degree + 2) // 2


def _generate_uniform_random_complex(length, key=None, seed=None):
    """
    In the unit disc centered at the origin.
    """
    if key is None:
        if seed is None:
            seed = random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    rnd1 = jax.random.uniform(
        key=subkey, shape=(length,), dtype=np.float64, minval=0, maxval=1
    )
    key, subkey = jax.random.split(key)
    rnd2 = jax.random.uniform(
        key=subkey, shape=(length,), dtype=np.float64, minval=0, maxval=2 * np.pi
    )

    return subkey, np.sqrt(rnd1) * np.exp(1.0j * rnd2)


def _calculate_all_normal_odering_terms(cutoff, degree):
    terms = []

    if degree == 0:
        return terms

    for i in range(degree):
        current_degree_terms = _calculate_normal_ordering_term_of_degree(cutoff, i + 1)
        terms.append(current_degree_terms)

    terms = list(itertools.chain.from_iterable(terms))
    # Due to the order of coefficient generation.
    # This way the first elements are the highest degree, and decreasing
    terms.reverse()

    return terms


def _calculate_normal_ordering_term_of_degree(cutoff, degree):
    creation_op = get_creation_operator(cutoff)
    annihilation_op = creation_op.T

    terms = []

    for i in range(degree + 1):
        creation_power = np.linalg.matrix_power(creation_op, degree - i)
        annihilation_power = np.linalg.matrix_power(annihilation_op, i)
        terms.append(np.matmul(creation_power, annihilation_power))

    return terms


def _generate_random_coefficients_of_degree(key, degree):
    if degree == 0:
        return [jax.random.uniform(key, dtype=np.float64, minval=0, maxval=1)]

    poly = []

    if degree % 2 == 0:

        key, complex_coeffs = _generate_uniform_random_complex(
            length=degree // 2, key=key
        )
        complex_coeffs = complex_coeffs.tolist()

        conj_coeffs = [np.conj(complex_coeffs[i]) for i in range(degree // 2)]
        conj_coeffs.reverse()

        real_coeff = [jax.random.uniform(key, dtype=np.float64, minval=0, maxval=1)]
        key, _ = jax.random.split(key)
        poly += complex_coeffs + real_coeff + conj_coeffs

    else:

        key, complex_coeffs = _generate_uniform_random_complex(
            length=(degree // 2) + 1, key=key
        )
        complex_coeffs = complex_coeffs.tolist()

        conj_coeffs = [np.conj(complex_coeffs[i]) for i in range((degree // 2) + 1)]
        conj_coeffs.reverse()

        poly += complex_coeffs + conj_coeffs

    return poly


def _generate_all_random_coefficients(key, degree):
    """
    NOTE: Coefficients are ordered in a descending manner regarding the degree of terms.
            For each degree of normally ordered terms the first term is the highest creation operator.
            Example:
            Let 'a' be the annihilation operator, and 'b' be the creation operator, for degree 3.
            Then b^3 + b^2a + ba^2 + a^3 + b^2 + ba + a^2 + b + a + 1 is the order of elements here.
    """
    if degree < 0:
        return []

    coeff = _generate_random_coefficients_of_degree(key, degree)
    key, _ = jax.random.split(key)
    return coeff + _generate_all_random_coefficients(key, degree - 1)


def _calculate_hamiltonian(key, degree, cutoff, coefficients=None):
    if coefficients is None:
        coefficients = _generate_all_random_coefficients(key, degree)

    terms = _calculate_all_normal_odering_terms(cutoff, degree)

    result = np.identity(cutoff, dtype=np.complex128)
    result = result * coefficients[-1]  # Constant term

    for i in range(len(coefficients) - 1):
        result += coefficients[i] * terms[i]

    return result


def generate_random_unitary(key, degree, cutoff, param, coefficients=None):
    hamiltonian = _calculate_hamiltonian(key, degree, cutoff, coefficients)
    unitary = get_unitary(param, hamiltonian)

    assert np.allclose(np.identity(cutoff), unitary @ np.conj(unitary).T)
    return unitary
