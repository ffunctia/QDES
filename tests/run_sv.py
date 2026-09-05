import sys
import time
import pickle
import resource
from qdes.gates import make_gate
from qdes.sv_simulator_gf import StateVectorQuantumSystem

def main():
    pickle_file = sys.argv[1]
    n_qubits = int(sys.argv[2])

    with open(pickle_file, 'rb') as f:
        execution_plan = pickle.load(f)

    start_time = time.perf_counter()
    sv = StateVectorQuantumSystem(n_qubits)

    for step in execution_plan:
        if step[0] == "GATE":
            sv.apply(make_gate(step[1], *step[2]), step[3])
        elif step[0] == "MEASURE":
            _ = sv.measure(step[1])

    _ = sv.sample(count=16384)

    speed = time.perf_counter() - start_time
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(f"{speed:.6f},{peak_mem_mb:.2f}")
if __name__ == "__main__":
    main()
