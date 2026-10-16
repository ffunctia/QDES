import random
import itertools
from numpy import asarray, array, zeros
import quimb.tensor as qtn

from . import gates as g
from . import utils_cside
from .gates import make_gate
from .simulator import QuantumSystem
from .sv_simulator import StateVectorQuantumSystem

from .constants import (
    MAX_QUBITS,
    MAX_STEPS,

    CPP_FACTOR_ENTANGLED_EPS,
    CPP_ENTANGLED_TOL,
    CPP_ENTANGLED_CHUNK,
    DTYPE_INT,
    CPP_NORM_EPS,
    RANDOM_PARAM_SINGLE_QUBIT_GATES,
    RANDOM_PARAM_TWO_QUBIT_GATES,
    RANDOM_OPERATION_PROBABILITIES,
    RANDOM_SINGLE_FIXED_NAMES,
    RANDOM_TWO_FIXED_NAMES,
)

def rand_angle():
    return random.uniform(0, 2 * 3.141592653589793)
#+
def _random_gate_and_meta(k, is_parametric):
    if k == 1:
        pool = RANDOM_PARAM_SINGLE_QUBIT_GATES if is_parametric else RANDOM_SINGLE_FIXED_NAMES
    elif k == 2:
        pool = RANDOM_PARAM_TWO_QUBIT_GATES if is_parametric else RANDOM_TWO_FIXED_NAMES
    else:
        raise ValueError("Only 1 and 2-qubit gates are supported in tests.")

    p_type = random.choice(pool)

    if is_parametric:
        if p_type == 'U3':
            a0, a1, a2 = rand_angle(), rand_angle(), rand_angle()
            op_mat = getattr(g, p_type)(a0, a1, a2)
            return p_type, (a0, a1, a2), op_mat
        else:
            a0 = rand_angle()
            op_mat = getattr(g, p_type)(a0)
            return p_type, (a0,), op_mat
    else:
        op_mat = getattr(g, p_type)
        return p_type, (), op_mat
#+
def generate_circuit(n_qubits=MAX_QUBITS, max_steps=MAX_STEPS):
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2.")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")

    execution_plan = []
    
    weights_to_use = RANDOM_OPERATION_PROBABILITIES

    for _ in range(max_steps):
        operation_type = random.choices(
            range(5), weights=weights_to_use
        )[0]
        if operation_type == 0:
            execution_plan.append(("MEASURE", random.randrange(n_qubits)))
            continue

        n_targets = 1 if operation_type in [1, 2] else 2
        is_parametric = operation_type in [2, 4]
        targets = random.sample(range(n_qubits), n_targets)
        op_name, params, matrix = _random_gate_and_meta(n_targets, is_parametric)
        execution_plan.append(("GATE", op_name, params, targets))

    return execution_plan
#+
def run_on_custom(ops, n_qubits):
    qs = QuantumSystem(n_qubits)
    for step in ops:
        if step[0] == "GATE":
            _, op_name, params, targets = step
            qs.apply(op_name, targets, *params)
        else:
            qs.measure(step[1])
    return qs
#+
def run_on_quimb(ops, n_qubits):
    circ = qtn.Circuit(n_qubits)
    for step in ops:
        if step[0] == "GATE":
            _, op_name, params, targets = step
            circ.apply_gate_raw(make_gate(op_name, *params), targets)
    sv = asarray(circ.to_dense()).flatten()
    return sv
#+
def run_on_sv_sim(ops, n_qubits):
    qs = StateVectorQuantumSystem(n_qubits)
    for step in ops:
        if step[0] == "GATE":
            _, op_name, params, targets = step
            qs.apply(op_name, targets, *params)
        else:
            qs.measure(step[1])
    return qs.state
#+
def project_qubit(sv_flat, n, qubit, outcome):
    sv_copy = sv_flat.copy()
    utils_cside.collapse_and_normalize(sv_copy, qubit, outcome, 1.0, CPP_NORM_EPS)
    
    return sv_copy
#+
def find_true_blocks_cpp(sv_flat, n):
    remaining = list(range(n))
    blocks = []

    while remaining:
        i = remaining[0]
        others = [q for q in remaining if q != i]
        found = None

        for size in range(1, len(remaining) + 1):
            if size == 1:
                candidates = [(i,)]
            else:
                candidates = [(i,) + c for c in itertools.combinations(others, size - 1)]

            for subset in candidates:
                complement = tuple(q for q in range(n) if q not in subset)
                
                subset_cpp = array(subset, dtype=DTYPE_INT)
                comp_cpp = array(complement, dtype=DTYPE_INT)
                
                is_entangled = utils_cside.is_entangled_no_copy(
                    sv_flat, subset_cpp, comp_cpp, 
                    CPP_FACTOR_ENTANGLED_EPS, CPP_ENTANGLED_TOL, CPP_ENTANGLED_CHUNK
                )
                
                if not is_entangled:
                    found = subset
                    break
                    
            if found is not None:
                break

        if found is None:
            found = tuple(remaining)

        blocks.append(sorted(found))
        remaining = [q for q in remaining if q not in found]

    return blocks
#+
def _embed_gate(gate, gate_targets, active_targets):
    target_positions = [active_targets.index(q) for q in gate_targets]
    active_size = 2 ** len(active_targets)
    target_size = 2 ** len(gate_targets)
    embedded = zeros((active_size, active_size), dtype=gate.dtype)

    for active_input in range(active_size):
        target_input = 0
        for position in target_positions:
            target_input = (target_input << 1) | ((active_input >> (len(active_targets) - 1 - position)) & 1)

        for target_output in range(target_size):
            active_output = active_input
            for offset, position in enumerate(target_positions):
                bit = (target_output >> (len(gate_targets) - 1 - offset)) & 1
                mask = 1 << (len(active_targets) - 1 - position)
                active_output = (active_output & ~mask) | (bit * mask)
            embedded[active_output, active_input] = gate[target_output, target_input]

    return embedded
#+
def true_group_sizes(sv, n):
    blocks = find_true_blocks_cpp(sv, n)
    return sorted(len(b) for b in blocks)