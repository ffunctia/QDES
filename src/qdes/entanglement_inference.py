import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

from .gates import OPERATION_IDS
from .constants import MAX_QUBITS, MAX_STEPS

class EntanglementInference:
    def __init__(self, onnx_path=None, max_qubits=MAX_QUBITS, max_steps=MAX_STEPS):
        self.max_qubits = max_qubits
        self.max_steps = max_steps
        if onnx_path is None:
            onnx_path = self._find_model_path()
        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        self.triu_rows, self.triu_cols = np.triu_indices(max_qubits)

    @staticmethod
    def _find_model_path():
        configured_path = os.environ.get("QDES_MODEL_PATH")
        candidates = []
        if configured_path:
            candidates.append(Path(configured_path))
        candidates.append(Path.cwd() / "model_files" / "entanglement_model.onnx")
        candidates.append(Path(__file__).resolve().parents[2] / "model_files" / "entanglement_model.onnx")

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0] if configured_path else candidates[1]

    def predict(self, history):
        # Build input matrix (1, max_steps, 5)
        input_matrix = np.full((1, self.max_steps, 5), -1.0, dtype=np.float32)
        input_matrix[:, :, 3:] = 0.0

        start = max(0, len(history) - self.max_steps)
        for i, (q0, q1, op_name, sp, ep) in enumerate(history[start:]):
            op_id = OPERATION_IDS.get(op_name, -1)
            input_matrix[0, i] = [q0, q1, op_id, sp, ep]

        # Prepare inputs for ONNX
        q0 = (input_matrix[:, :, 0].astype(np.int64) + 1)
        q1 = (input_matrix[:, :, 1].astype(np.int64) + 1)
        op_id = (input_matrix[:, :, 2].astype(np.int64) + 1)
        params = input_matrix[:, :, 3:5].astype(np.float32)
        mask = (op_id != 0).astype(np.float32)

        # Run inference
        inputs = {
            'q0': q0, 'q1': q1, 'op_id': op_id,
            'params': params, 'mask': mask
        }
        logits = self.session.run(['logits'], inputs)[0]

        # Sigmoid and reshape to matrix
        with np.errstate(over='ignore'):
            probs = 1 / (1 + np.exp(-logits[0]))
        prob_matrix = np.zeros((self.max_qubits, self.max_qubits), dtype=np.float32)
        prob_matrix[self.triu_rows, self.triu_cols] = probs

        # Symmetrize
        for i in range(self.max_qubits):
            for j in range(i + 1, self.max_qubits):
                prob_matrix[j, i] = prob_matrix[i, j]
        return prob_matrix
