import sys
import time
import pickle
import resource
import quimb.tensor as qtn

from qdes.gates import make_gate, P0, P1

#+
def main():
    pickle_file = sys.argv[1]
    n_qubits = int(sys.argv[2])

    with open(pickle_file, 'rb') as f:
        execution_plan = pickle.load(f)

    start_time = time.perf_counter()
    circ = qtn.Circuit(n_qubits)

    for step in execution_plan:
        if step[0] == "GATE":
            circ.apply_gate_raw(make_gate(step[1], *step[2]), step[3])
        elif step[0] == "MEASURE":
            q = step[1]
            sample_res = next(circ.sample(1, qubits=[q]))
            
            outcome = int(sample_res)
            
            proj = P0 if outcome == 0 else P1
            circ.apply_gate_raw(proj, [q])

    _ = circ.sample(C = 16384)

    speed = time.perf_counter() - start_time
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(f"{speed:.6f},{peak_mem_mb:.2f}")

if __name__ == "__main__":
    main()