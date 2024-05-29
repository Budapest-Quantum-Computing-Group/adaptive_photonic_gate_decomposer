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

from enum import Enum
import jax.numpy as np

from src.circuits import GateIdx, evaluate_circuit
from src.optimization import avg_gate_fidelity, process_fidelity


class Stat(Enum):
    SEED = "seed"
    CUTOFF = "cutoff"
    GATE_CUTOFF = "gate_cutoff"
    NO_LAYER = "number_of_layers"
    NO_ITER = "number_of_weight_iteration"
    LR = "learning_rate"
    TAU = "parameter_threshold"

    PARAM = "target_gate_parameter"
    T_GATE = "target_gate"

    WEIGHTS = "weights"
    CIRC = "circuit_indices"
    L_GATE = "learnt_gate"

    PFID = "process_fidelity"
    AVGFID = "average_fidelity"

    MAXSQPARAM = "maximum_abs_value_sq_parameter"
    MAXDPPARAM = "maximum_abs_value_dp_parameter"
    MAXPSPARAM = "maximum_abs_value_ps_parameter"
    MAXKRPARAM = "maximum_abs_value_kr_parameter"

    MSG = "msg"


def get_stats(
    seed,
    cutoff,
    gate_cutoff,
    no_layer,
    weights,
    circuit,
    target_param,
    target_gate,
    param_threshold,
    learning_rate,
    no_weight_iteration,
):
    stats = {}
    stats[Stat.SEED] = seed
    stats[Stat.CUTOFF] = cutoff
    stats[Stat.GATE_CUTOFF] = gate_cutoff
    stats[Stat.NO_LAYER] = no_layer
    stats[Stat.WEIGHTS] = weights
    stats[Stat.CIRC] = circuit
    stats[Stat.NO_ITER] = no_weight_iteration
    stats[Stat.LR] = learning_rate
    stats[Stat.TAU] = param_threshold
    stats[Stat.PARAM] = target_param
    stats[Stat.T_GATE] = target_gate

    if len(weights) == 0:
        stats[Stat.L_GATE] = np.zeros(target_gate.shape)
        stats[Stat.PFID] = 0
        stats[Stat.AVGFID] = 0
        stats[Stat.MAXPSPARAM] = 0
        stats[Stat.MAXSQPARAM] = 0
        stats[Stat.MAXDPPARAM] = 0
        stats[Stat.MAXKRPARAM] = 0
        stats[Stat.MSG] = "Empty circuit, possibly every weight was filtered out."
        return

    learnt_gate = evaluate_circuit(weights, circuit, cutoff)

    proc_fid = process_fidelity(target_gate, learnt_gate, cutoff, gate_cutoff)
    avg_fid = avg_gate_fidelity(target_gate, learnt_gate, gate_cutoff)

    max_ps = np.max(np.abs(weights[circuit == GateIdx.PS.value]), initial=0)
    max_dp = np.max(np.abs(weights[circuit == GateIdx.DP.value]), initial=0)
    max_sq = np.max(np.abs(weights[circuit == GateIdx.SQ.value]), initial=0)
    max_kr = np.max(np.abs(weights[circuit == GateIdx.KR.value]), initial=0)

    stats[Stat.L_GATE] = learnt_gate
    stats[Stat.PFID] = proc_fid
    stats[Stat.AVGFID] = avg_fid
    stats[Stat.MAXPSPARAM] = max_ps
    stats[Stat.MAXSQPARAM] = max_sq
    stats[Stat.MAXDPPARAM] = max_dp
    stats[Stat.MAXKRPARAM] = max_kr
    stats[Stat.MSG] = "Success"

    return stats
