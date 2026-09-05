import numpy as np
from .utils_cside import (
    apply_gate,
    measure_probability,
    collapse_and_normalize,
)
from .constants import (
    DTYPE_COMPLEX,
    DTYPE_INT,
    CPP_NORM_EPS,
    RAND_SEED,
)

class StateVectorQuantumSystem:
    #+
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=DTYPE_COMPLEX)
        self.state[0] = 1.0 + 0j
    #+
    def apply(self, gate, target_qubits):
        logical_indices = np.array(target_qubits, dtype=DTYPE_INT)
        apply_gate(self.state, gate, logical_indices)
    #+
    def measure(self, qubit):
        p1 = measure_probability(self.state, qubit)
        p1 = min(max(p1, 0.0), 1.0)
        p0 = 1.0 - p1

        result = int(np.random.choice([0, 1], p=[p0, p1]))
        norm_value = p1 if result == 1 else p0

        collapse_and_normalize(
            self.state, qubit, result, norm_value, CPP_NORM_EPS
        )

        return result
    #+
    def sample(self, target_qubits=None, count=1000, return_counts=True, seed=None):
        if target_qubits is None:
            target_qubits = list(range(self.n))
        elif isinstance(target_qubits, int):
            target_qubits = [target_qubits]

        if seed is None:
            seed = RAND_SEED()

        logical_indices = np.array(target_qubits, dtype=DTYPE_INT)

        from .utils_cside import sample_group_shots
        samples = sample_group_shots(
            self.state, logical_indices, count, seed, CPP_NORM_EPS
        )
        
        samples = samples.astype(np.uint64)
        k_total = len(target_qubits)

        if return_counts:
            unique, counts = np.unique(samples, return_counts=True)
            fmt = f"0{k_total}b"
            return {format(val, fmt): int(cnt) for val, cnt in zip(unique, counts)}
        
        return samples
    #+
    def full_state(self):
        sv = self.state.view()
        sv.flags.writeable = False
        return sv, list(range(self.n))