import sys
import time
import pickle
import resource
from qdes.simulator import QuantumSystem

#+
def main():
    pickle_file = sys.argv[1]
    n_qubits = int(sys.argv[2])

    with open(pickle_file, 'rb') as f:
        execution_plan = pickle.load(f)

    start_time = time.perf_counter()
    qs = QuantumSystem(n_qubits)

    for step in execution_plan:
        if step[0] == "GATE": qs.apply(step[1], step[3], *step[2])
        elif step[0] == "MEASURE": _ = qs.measure(step[1])

    _ = qs.sample(count = 16384)

    speed = time.perf_counter() - start_time
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(f"{speed:.6f},{peak_mem_mb:.2f}")

if __name__ == "__main__":
    main()