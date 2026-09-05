import sys
import random
import numpy as np
import pickle
import os

from qdes.constants import MAX_QUBITS, CALCULATE_STEPS
from qdes.simulator import QuantumSystem
from qdes.utils_pyside import generate_circuit
from qdes.gates import compute_sp, compute_ep, make_gate

#+
def calculate_er(group_sizes, num_qubits):
    return sum((size / num_qubits) ** 2 for size in group_sizes if size > 1)

#+
def calculate_circuit_metrics(ops, execution_plan, n_qubits, n_measure):
    multi_gate_distances = []
    n_multi_gates = 0
    
    for _U, targets, _op_name, _sp, _ep in ops:
        if len(targets) >= 2:
            n_multi_gates += 1
            multi_gate_distances.append(abs(targets[0] - targets[1]))

    avg_gate_distance = (
        sum(multi_gate_distances) / len(multi_gate_distances)
        if multi_gate_distances else 0.0
    )

    qubit_depth = [0] * n_qubits
    for step in execution_plan:
        if step[0] == "GATE":
            targets = step[3]
            current_max = max((qubit_depth[q] for q in targets), default=0)
            new_depth = current_max + 1
            for q in targets:
                qubit_depth[q] = new_depth
    max_depth = max(qubit_depth) if qubit_depth else 0

    return {
        "avg_gate_distance": avg_gate_distance,
        "n_multi_gates": n_multi_gates,
        "max_depth": max_depth,
        "n_measurements": n_measure,
    }

#+
def generate_single_test(test_no, n_qubits, output_dir):
    if n_qubits > MAX_QUBITS:
        sys.stderr.write(f"Unsupported qubit count: {n_qubits}")
        sys.exit(1)

    seed = 1000 + test_no
    random.seed(seed)
    np.random.seed(seed)

    max_allowed_steps = CALCULATE_STEPS(n_qubits)
    execution_plan = generate_circuit(n_qubits, max_allowed_steps)
    ops = [
        (
            make_gate(step[1], *step[2]), step[3], step[1],
            compute_sp(step[1], *step[2]), compute_ep(step[1], *step[2])
        )
        for step in execution_plan if step[0] == "GATE"
    ]
    n_gates = len(ops)
    n_measure = sum(step[0] == "MEASURE" for step in execution_plan)

    qs_dummy = QuantumSystem(n_qubits)
    er_history = []
    bm_er_history = []

    for step in execution_plan:
        if step[0] == "MEASURE":
            bm_er_history.append(calculate_er(qs_dummy.custom_group_sizes(), n_qubits))
            qs_dummy.measure(step[1])
            er_history.append(calculate_er(qs_dummy.custom_group_sizes(), n_qubits))
        elif step[0] == "GATE":
            qs_dummy.apply(step[1], step[3], *step[2])
            er_history.append(calculate_er(qs_dummy.custom_group_sizes(), n_qubits))

    max_er = max(er_history) if er_history else 0.0
    avg_er = sum(er_history) / len(er_history) if er_history else 0.0
    avg_bm_er = sum(bm_er_history) / len(bm_er_history) if bm_er_history else 0.0

    del qs_dummy

    metrics = calculate_circuit_metrics(ops, execution_plan, n_qubits, n_measure)

    formatted_test_no = f"{test_no:03d}"
    os.makedirs(output_dir, exist_ok=True)
    pickle_filename = os.path.join(
        output_dir, f"test_{formatted_test_no}_{n_qubits}q.pkl"
    )
    
    with open(pickle_filename, 'wb') as f:
        pickle.dump(execution_plan, f)

    print(
        f"{test_no},{n_qubits},{n_gates},{max_er:.6f},{avg_er:.6f},{avg_bm_er:.6f},"
        f"{metrics['avg_gate_distance']:.6f},{metrics['n_multi_gates']},"
        f"{metrics['max_depth']},{metrics['n_measurements']}"
    )
    sys.stderr.write(f"Test {formatted_test_no} generated: {pickle_filename}\n")

def main():
    if len(sys.argv) < 4:
        print("Usage: python controller.py <test_no> <n_qubits> <output_dir>")
        sys.exit(1)
        
    test_no = int(sys.argv[1])
    n_qubits = int(sys.argv[2])
    output_dir = sys.argv[3]
    generate_single_test(test_no, n_qubits, output_dir)

if __name__ == "__main__":
    main()