import random
import sys

import numpy as np
import quimb.tensor as qtn

from qdes.constants import TEST_FIDELITY_TOL, DEFAULT_TEST_SEED
from qdes.gates import make_gate
from qdes.sv_simulator import StateVectorQuantumSystem
from qdes.utils_pyside import _random_gate_and_meta

#+
def generate_gate_only_circuit(n_qubits, n_steps, seed):
    random.seed(seed)
    rng = random.Random(seed)
    circuit = []
    for _ in range(n_steps):
        n_targets = 1 if rng.random() < 0.5 else 2
        targets = rng.sample(range(n_qubits), n_targets)
        is_parametric = rng.random() < 0.5
        name, params, _ = _random_gate_and_meta(n_targets, is_parametric)
        circuit.append((name, params, targets))
    return circuit

#+
def run_sv(circuit, n_qubits):
    simulator = StateVectorQuantumSystem(n_qubits)
    for name, params, targets in circuit:
        simulator.apply(make_gate(name, *params), targets)
    return simulator.state

#+
def run_quimb(circuit, n_qubits):
    simulator = qtn.Circuit(n_qubits)
    for name, params, targets in circuit:
        simulator.apply_gate_raw(make_gate(name, *params), targets)
    return np.asarray(simulator.to_dense()).flatten()

#+
def compute_fidelity(left, right):
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(abs(np.vdot(left, right)) ** 2)

#+
def main(seed=None):
    n_qubits = 8
    n_steps = 122
    first_seed = seed
    n_tests = 1000
    failures = []

    for test_number in range(n_tests):
        seed = first_seed + test_number
        circuit = generate_gate_only_circuit(n_qubits, n_steps, seed)
        sv_state = run_sv(circuit, n_qubits)
        quimb_state = run_quimb(circuit, n_qubits)
        fidelity = compute_fidelity(sv_state, quimb_state)
        max_error = float(np.max(np.abs(sv_state - quimb_state)))

        if abs(1.0 - fidelity) >= TEST_FIDELITY_TOL:
            failures.append((test_number + 1, seed, fidelity, max_error))

        print(
            f"[{test_number + 1:03d}/{n_tests}] seed={seed} "
            f"fidelity={fidelity:.16f} max_error={max_error:.3e}"
        )

    passed = n_tests - len(failures)
    print(f"SUMMARY: passed={passed}/{n_tests}, failed={len(failures)}")
    if failures:
        raise AssertionError(f"Fidelity failures: {failures}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEST_SEED
    main(seed=seed)