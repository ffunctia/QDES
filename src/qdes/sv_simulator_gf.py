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
    MAX_FUSION_QUBITS 
)
from .gates import make_gate
#+
def _embed_gate(gate, gate_targets, active_targets):
    target_positions = [active_targets.index(q) for q in gate_targets]
    active_size = 2 ** len(active_targets)
    target_size = 2 ** len(gate_targets)
    embedded = np.zeros((active_size, active_size), dtype=gate.dtype)

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

class StateVectorQuantumSystem:
    #+
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=DTYPE_COMPLEX)
        self.state[0] = 1.0 + 0j
        self._fusion_ops = [] 
    #+
    def _can_queue_gate(self, target_qubits):
        if not self._fusion_ops:
            return True
        
        active_targets = set(target_qubits)
        for _, queued_targets in self._fusion_ops:
            active_targets.update(queued_targets)
        return len(active_targets) <= MAX_FUSION_QUBITS
    #+
    def _flush_fusion(self):
        if not self._fusion_ops:
            return
            
        active_targets = []
        for _, targets in self._fusion_ops:
            for qubit in targets:
                if qubit not in active_targets:
                    active_targets.append(qubit)

        fused_gate = np.eye(2 ** len(active_targets), dtype=self._fusion_ops[0][0].dtype)
        for gate, targets in self._fusion_ops:
            embedded_gate = _embed_gate(gate, targets, active_targets)
            fused_gate = embedded_gate @ fused_gate

        logical_indices = np.array(active_targets, dtype=DTYPE_INT)
        apply_gate(self.state, fused_gate, logical_indices)
        
        self._fusion_ops.clear()
    #+
    def apply(self, gate, target_qubits):
        target_qubits_tuple = tuple(target_qubits)
        if self._can_queue_gate(target_qubits_tuple):
            self._fusion_ops.append((gate, target_qubits_tuple))
            return

        self._flush_fusion()
        logical_indices = np.array(target_qubits_tuple, dtype=DTYPE_INT)
        apply_gate(self.state, gate, logical_indices)
    #+
    def measure(self, qubit):
        self._flush_fusion() 
        
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
        self._flush_fusion() 
        
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
        self._flush_fusion() 
        sv = self.state.view()
        sv.flags.writeable = False
        return sv, list(range(self.n))