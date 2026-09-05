from itertools import combinations
import numpy as np

from .utils_cside import (
    apply_gate, kron_merge, is_entangled_no_copy, factor_rank1_no_copy,
    measure_probability, collapse_and_normalize, collapse_and_extract
)
from .constants import (
    MAX_QUBITS,
    DTYPE_INT,
    STATE_ZERO,
    STATE_ONE,
    THRESHOLD_ENTANGLED,
    THRESHOLD_NOT_ENTANGLED,
    BRANCH_DEPTH,
    CPP_FACTOR_ENTANGLED_EPS,
    CPP_ENTANGLED_TOL,
    CPP_ENTANGLED_CHUNK,
    CPP_NORM_EPS,
    MAX_FUSION_QUBITS,
    RAND_SEED
)
from .gates import make_gate, compute_sp, compute_ep

_inference = None
_INFERENCE_LOADED = False
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

#+
def _get_inference():
    global _inference, _INFERENCE_LOADED
    if not _INFERENCE_LOADED:
        try:
            from .entanglement_inference import EntanglementInference
            _inference = EntanglementInference()
        except Exception as e:
            print(f"Onnx model is not available: {e}")
            _inference = None
        _INFERENCE_LOADED = True
    return _inference

class QuantumSystem:
    def __init__(self, n_qubits):
            
        self.n = min(n_qubits, MAX_QUBITS)         
        self.groups = {}          
        self.qubit_to_group = {}  
        self.group_qubits = {}    
        self._next_gid = 0        
        self.gate_history = []
        self._fusion_ops = {}

        for q in range(n_qubits):
            gid = self._new_group()
            self.groups[gid] = STATE_ZERO.copy()
            self.qubit_to_group[q] = gid
            self.group_qubits[gid] = [q]
    #+
    def _new_group(self):
        gid = self._next_gid
        self._next_gid += 1
        return gid
    #+
    def apply(self, gate, target_qubits, *params, op_name=None, sp=None, ep=None):
        if isinstance(gate, str):
            op_name = gate
            gate = make_gate(op_name, *params)
            sp = compute_sp(op_name, *params)
            ep = compute_ep(op_name, *params)
        else:
            op_name = op_name or "UNKNOWN"
            sp = 0.0 if sp is None else sp
            ep = 0.0 if ep is None else ep

        q0 = target_qubits[0]
        q1 = target_qubits[1] if len(target_qubits) > 1 else q0
        self.gate_history.append((q0, q1, op_name, sp, ep))

        target_qubits = tuple(target_qubits)
        if self._can_queue_gate(target_qubits):
            gid = self.qubit_to_group[target_qubits[0]]
            if gid not in self._fusion_ops:
                self._fusion_ops[gid] = []
            self._fusion_ops[gid].append((gate, target_qubits))
            return

        gids_involved = list(dict.fromkeys(self.qubit_to_group[q] for q in target_qubits))
        self._flush_fusion(gid_list=gids_involved)
        self._execute_gate(gate, target_qubits)
    #+
    def _can_queue_gate(self, target_qubits):
        if len(set(self.qubit_to_group[q] for q in target_qubits)) != 1:
            return False
        
        gid = self.qubit_to_group[target_qubits[0]]
        
        if gid not in self._fusion_ops:
            return True
        
        active_targets = set(target_qubits)
        for _, queued_targets in self._fusion_ops[gid]:
            active_targets.update(queued_targets)
        return len(active_targets) <= MAX_FUSION_QUBITS
    #+
    def _execute_gate(self, gate, target_qubits):
        k = len(target_qubits)
        if k == 1:
            self._apply_single(gate, target_qubits[0])
            return

        gids = [self.qubit_to_group[q] for q in target_qubits]
        if len(set(gids)) == 1:
            self._apply_within_group(gate, target_qubits, gids[0])
            return

        self._apply_across_groups(gate, target_qubits)
    #+
    def _flush_fusion(self, gid_list=None):
        if not self._fusion_ops:
            return

        if gid_list is None:
            gid_list = list(self._fusion_ops.keys())

        for gid in gid_list:
            if gid not in self._fusion_ops:
                continue
                
            ops = self._fusion_ops[gid]
            if not ops:
                continue
                
            active_targets = []
            for _, targets in ops:
                for qubit in targets:
                    if qubit not in active_targets:
                        active_targets.append(qubit)

            fused_gate = np.eye(2 ** len(active_targets), dtype=ops[0][0].dtype)
            for gate, targets in ops:
                embedded_gate = _embed_gate(gate, targets, active_targets)
                fused_gate = embedded_gate @ fused_gate

            group_qubits = self.group_qubits[gid]
            local_indices = np.array(
                [group_qubits.index(q) for q in active_targets],
                dtype=DTYPE_INT
            )
            apply_gate(self.groups[gid], fused_gate, local_indices)
            if len(group_qubits) > 1:
                self._resplit_dispatch(gid, merge_components=None)
            
            del self._fusion_ops[gid]
    #+
    def _apply_single(self, gate, qubit):
        gid = self.qubit_to_group[qubit]            
        sv = self.groups[gid]                       
        qubits_in_group = self.group_qubits[gid]    
        n_total = int(np.log2(len(sv)))             
        local_idx = qubits_in_group.index(qubit)    
        apply_gate(sv, gate, np.array([local_idx], dtype=DTYPE_INT))
    #+
    def _apply_within_group(self, gate, target_qubits, gid):
        sv = self.groups[gid]
        qubits_in_group = self.group_qubits[gid] 
        n_total = int(np.log2(len(sv))) 
        local_indices = np.array(
            [qubits_in_group.index(q) for q in target_qubits],
            dtype=DTYPE_INT
        )
        apply_gate(sv, gate, local_indices)
        if n_total > 1:
            del sv
            self._resplit_dispatch(gid, merge_components=None)
    #+
    def _apply_across_groups(self, gate, target_qubits):
        gids_involved = list(dict.fromkeys(self.qubit_to_group[q] for q in target_qubits))

        if len(gids_involved) == 1:
            merged_gid = gids_involved[0]
            merged_qubits = self.group_qubits[merged_gid]
            merge_components = None
        else:
            merge_components = [list(self.group_qubits[g]) for g in gids_involved]
            merged_qubits, merged_gid = self._merge_groups(gids_involved)

        sv = self.groups[merged_gid]
        n_total = int(np.log2(len(sv)))
        local_indices = np.array(
            [merged_qubits.index(q) for q in target_qubits],
            dtype=DTYPE_INT
        )
        apply_gate(sv, gate, local_indices)
        del sv
        self._resplit_dispatch(merged_gid, merge_components=merge_components)
    #+
    def _merge_groups(self, gids): 
        merged_sv = self.groups.pop(gids[0])
        merged_qubits = list(self.group_qubits.pop(gids[0]))

        for gid in gids[1:]:
            next_sv = self.groups.pop(gid)
            merged_sv = kron_merge(merged_sv, next_sv)
            merged_qubits += self.group_qubits.pop(gid)

        new_gid = self._new_group()
        self.groups[new_gid] = merged_sv 
        self.group_qubits[new_gid] = merged_qubits
        
        for q in merged_qubits:
            self.qubit_to_group[q] = new_gid

        return merged_qubits, new_gid
    #+
    def _get_ml_predictions(self):
        if len(self.gate_history) == 0:
            return None
        inf = _get_inference()
        if inf is None:
            return None
        try:
            return inf.predict(self.gate_history)
        except Exception:
            return None
    #+
    def _resplit_dispatch(self, gid, merge_components=None):
        sv = self.groups[gid] 
        qubits = self.group_qubits[gid]
        m = len(qubits)
        if m <= 1: return

        n_total = m
        qubits_set = set(qubits) 
        found_partition = None
        tried_qs = set()

        def _try(cand_qs):
            cand_set = frozenset(cand_qs)
            if not cand_set or cand_set == qubits_set or cand_set in tried_qs:
                return None
            tried_qs.add(cand_set)
            left_qs = [q for q in qubits if q in cand_set]
            right_qs = [q for q in qubits if q not in cand_set]
            sub_idx = [qubits.index(q) for q in left_qs]
            comp_idx = [qubits.index(q) for q in right_qs]
            
            if not is_entangled_no_copy(sv, np.array(sub_idx, dtype=DTYPE_INT), np.array(comp_idx, dtype=DTYPE_INT),
                                        CPP_FACTOR_ENTANGLED_EPS, CPP_ENTANGLED_TOL, CPP_ENTANGLED_CHUNK):
                return (left_qs, right_qs, sub_idx, comp_idx)
            return None

        
        # 1. Older merge components approach
        if not found_partition and merge_components and len(merge_components) > 1:
            k = len(merge_components)
            for size in range(1, k):
                for combo in combinations(range(k), size):
                    cand = [q for i in combo for q in merge_components[i]]
                    found_partition = _try(cand)
                    if found_partition: break
                if found_partition: break

        # 2. ML based 3-stage approach
        if not found_partition:
            prob_matrix = self._get_ml_predictions()
            if prob_matrix is not None:
                def get_components(qs, threshold):
                    parent = {q: q for q in qs}

                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx != ry:
                            parent[ry] = rx

                    for i, qi in enumerate(qs):
                        for qj in qs[i+1:]:
                            if prob_matrix[qi, qj] > threshold:
                                union(qi, qj)

                    clusters = {}
                    for q in qs:
                        clusters.setdefault(find(q), []).append(q)
                    return list(clusters.values())

                # Stage 1: Not entangled
                comps_04 = get_components(qubits, THRESHOLD_NOT_ENTANGLED)
                if len(comps_04) > 1:
                    for comp in comps_04:
                        found_partition = _try(comp)
                        if found_partition: break

                # Stage 2 & 3: Entangled and then fallback to sure clusters
                if not found_partition:
                    sure_clusters = get_components(qubits, THRESHOLD_ENTANGLED)
                    if len(sure_clusters) > 1:
                        n_clusters = len(sure_clusters)
                        limit = min(BRANCH_DEPTH, n_clusters // 2)
                        if limit < 1: limit = 1
                        
                        for size in range(1, limit + 1):
                            if found_partition: break
                            for cluster_combo in combinations(sure_clusters, size):
                                cand_qs = [q for c in cluster_combo for q in c]
                                found_partition = _try(cand_qs)
                                if found_partition: break

        # Partitioning and new groups
        if found_partition:
            left_qs, right_qs, sub_idx, comp_idx = found_partition
            sv_a, sv_b = factor_rank1_no_copy(sv, np.array(sub_idx, dtype=DTYPE_INT), 
                                              np.array(comp_idx, dtype=DTYPE_INT), CPP_FACTOR_ENTANGLED_EPS)

            del sv
            del self.groups[gid]
            del self.group_qubits[gid]

            gid_a = self._new_group()
            self.groups[gid_a] = sv_a
            self.group_qubits[gid_a] = left_qs
            for q in left_qs: self.qubit_to_group[q] = gid_a

            gid_b = self._new_group()
            self.groups[gid_b] = sv_b
            self.group_qubits[gid_b] = right_qs
            for q in right_qs: self.qubit_to_group[q] = gid_b

            self._resplit_dispatch(gid_a)
            self._resplit_dispatch(gid_b)
    #+
    def measure(self, qubit):
        self._flush_fusion()
        self.gate_history.append((qubit, qubit, "MEASURE", -1.0, -1.0))

        gid = self.qubit_to_group[qubit]
        sv = self.groups[gid]
        qubits_in_group = self.group_qubits[gid]
        k = len(qubits_in_group)
        local_idx = qubits_in_group.index(qubit)
 
        p1 = measure_probability(sv, local_idx)
        p1 = min(max(p1, 0.0), 1.0)
        p0 = 1.0 - p1
 
        result = int(np.random.choice([0, 1], p=[p0, p1]))
        norm_value = p1 if result == 1 else p0
 
        if k == 1:
            collapse_and_normalize(sv, local_idx, result, norm_value, CPP_NORM_EPS)
        else:
            self._remove_qubit_from_group(qubit, result, gid, qubits_in_group, sv, norm_value)
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

        group_map = {}
        for idx, q in enumerate(target_qubits):
            gid = self.qubit_to_group[q]
            if gid not in group_map:
                group_map[gid] = []
            group_map[gid].append((idx, q))

        combined_samples = np.zeros(count, dtype=np.uint64)
        k_total = len(target_qubits)

        for gid, q_info in group_map.items():
            sv = self.groups[gid]
            qubits_in_group = self.group_qubits[gid]
            logical_indices = np.array(
                [qubits_in_group.index(q) for _, q in q_info],
                dtype=DTYPE_INT
            )

            from .utils_cside import sample_group_shots
            group_samples = sample_group_shots(
                sv, logical_indices, count, seed, CPP_NORM_EPS
            )

            k_g = len(q_info)
            for local_pos, (global_target_pos, _) in enumerate(q_info):
                bit_shift_in_group = k_g - 1 - local_pos
                bits = (group_samples.astype(np.uint64) >> bit_shift_in_group) & 1
                
                global_shift = k_total - 1 - global_target_pos
                combined_samples |= (bits << global_shift)

        if return_counts:
            unique, counts = np.unique(combined_samples, return_counts=True)
            fmt = f"0{k_total}b"
            return {format(val, fmt): int(cnt) for val, cnt in zip(unique, counts)}
        
        return combined_samples
    #+
    def _remove_qubit_from_group(self, qubit, result, gid, qubits_in_group, sv, norm_value):
        local_idx = qubits_in_group.index(qubit)
        qubit_sv = STATE_ZERO.copy() if result == 0 else STATE_ONE.copy()
        remaining_qubits = [q for q in qubits_in_group if q != qubit]
 
        remaining_sv = collapse_and_extract(
            sv, local_idx, result, norm_value, CPP_NORM_EPS
        )
 
        del sv
        del self.groups[gid]
        del self.group_qubits[gid]
 
        gid_a = self._new_group()
        self.groups[gid_a] = qubit_sv
        self.group_qubits[gid_a] = [qubit]
        self.qubit_to_group[qubit] = gid_a
 
        gid_b = self._new_group()
        self.groups[gid_b] = remaining_sv
        self.group_qubits[gid_b] = remaining_qubits
        for q in remaining_qubits: self.qubit_to_group[q] = gid_b

        if len(remaining_qubits) > 1:
            self._resplit_dispatch(gid_b)
    #+
    def custom_group_sizes(self):
        self._flush_fusion()
        return sorted(len(qubits) for qubits in self.group_qubits.values())
    #+
    def full_state(self):
        self._flush_fusion()
        keys = list(self.groups.keys())
        if len(keys) == 1:
            sv = self.groups[keys[0]].view()
            sv.flags.writeable = False
            return sv, list(self.group_qubits[keys[0]])

        sv = self.groups[keys[0]]
        qubits_order = list(self.group_qubits[keys[0]])
        for gid in keys[1:]:
            sv = kron_merge(sv, self.groups[gid])
            qubits_order += self.group_qubits[gid]
        return sv, qubits_order