import sys
import random
import numpy as np
from qdes.constants import (
    PYTHON_NORM_EPS,
    MAX_QUBITS,
    TEST_FIDELITY_TOL,
    CALCULATE_STEPS,
    DEFAULT_TEST_SEED
)
from qdes.simulator import QuantumSystem
from qdes.sv_simulator import StateVectorQuantumSystem
from qdes.utils_pyside import (
    generate_circuit,
    make_gate,
    project_qubit,
    true_group_sizes
)
#+
def compute_fidelity(sv_custom, sv_ref):
    inner_product = np.vdot(sv_ref, sv_custom)
    return float(np.abs(inner_product) ** 2)

#+
def reconstruct_full_statevector(qs, n_qubits):
    sv, qubits_order = qs.full_state()
    
    tensor = sv.reshape([2] * len(qubits_order))
    perm = [qubits_order.index(i) for i in range(n_qubits)]
    tensor = np.transpose(tensor, perm)
    
    return tensor.flatten()

def run_interleaved_sim(actions, n_qubits):
    qs = QuantumSystem(n_qubits)
    qs_ref = StateVectorQuantumSystem(n_qubits)
    
    outcomes = {}
    for action in actions:
        if action[0] == 'GATE':
            _, op_name, params, targets = action
            qs.apply(op_name, targets, *params)
            qs_ref.apply(make_gate(op_name, *params), targets)
            
        elif action[0] == 'MEASURE':
            _, q = action
            r = qs.measure(q)
            outcomes[q] = r
            
            sv_ref = qs_ref.state
            sv_ref = project_qubit(sv_ref, n_qubits, q, r)
            norm = np.linalg.norm(sv_ref)
            
            if norm < PYTHON_NORM_EPS:
                return qs, sv_ref, outcomes, False 
                
            sv_ref = sv_ref / norm
            qs_ref.state = sv_ref
            
    return qs, qs_ref.state, outcomes, True

#+
def run_test(n_tests, n_qubits=MAX_QUBITS // 2, n_gates_range=None, seed=None,
             tol=TEST_FIDELITY_TOL, verbose_fail=True):

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2.")

    score_fidelity = 0
    score_groups = 0
    score_both = 0
    fails = []
    skipped = 0

    print("Running interleaved simulator tests...")

    for t in range(n_tests):
        if n_gates_range is None:
            max_allowed_steps = CALCULATE_STEPS(n_qubits)
            current_steps_range = (n_qubits, max_allowed_steps)
        else:
            current_steps_range = n_gates_range
            
        max_steps = random.randint(*current_steps_range)
        
        steps = generate_circuit(n_qubits, max_steps)
        n_gates = sum(step[0] == "GATE" for step in steps)
        n_measure = sum(step[0] == "MEASURE" for step in steps)
        measure_order = [step[1] for step in steps if step[0] == "MEASURE"]

        qs, sv_ref, outcomes, ok_run = run_interleaved_sim(steps, n_qubits)
        if not ok_run:
            skipped += 1
            continue

        sv_custom = reconstruct_full_statevector(qs, n_qubits)
        sv_custom = sv_custom / np.linalg.norm(sv_custom)
        fidelity = compute_fidelity(sv_custom, sv_ref)
        ok_fidelity = abs(1.0 - fidelity) < tol

        custom_sizes = qs.custom_group_sizes()
        ref_sizes = true_group_sizes(sv_ref, n_qubits)
        ok_groups = (custom_sizes == ref_sizes)

        ok = ok_fidelity and ok_groups
        score_fidelity += int(ok_fidelity)
        score_groups += int(ok_groups)
        score_both += int(ok)

        status = "PASS" if ok else "FAIL"
        print(f"{status}! TEST: {t + 1}/{n_tests} steps={len(steps)} (gates={n_gates}, measure={n_measure}) fid={fidelity} score={score_both}/{t + 1 - skipped}")

        if not ok:
            fails.append({
                "test": t, "n_gates": n_gates, "measured": measure_order,
                "outcomes": outcomes, "fidelity": fidelity,
                "fidelity_ok": ok_fidelity, "custom_sizes": custom_sizes,
                "ref_sizes": ref_sizes, "groups_ok": ok_groups,
            })
            if verbose_fail:
                print(f"fid_ok={ok_fidelity}, groups_ok={ok_groups}")
                if not ok_groups:
                    print(f"            custom={custom_sizes} | ref={ref_sizes}")

    ran = n_tests - skipped
    print("TEST SUMMARY")
    print(f"Successful tests       : {score_both}/{ran} ({score_both / max(1, ran) * 100}%)")
    print(f"Fidelity               : {score_fidelity}/{ran}")
    print(f"Entanglement grouping  : {score_groups}/{ran}")
    if skipped > 0:
        print(f" Skipped tests          : {skipped} (zero-probability measurement)")

    return score_both, ran, fails

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TEST_SEED
    n_qubits = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_QUBITS // 2

    run_test(n, n_qubits=n_qubits, seed=seed)