import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from qdes.gates import OPERATION_IDS, NUM_OP_IDS, compute_sp, compute_ep, make_gate
from qdes.sv_simulator import StateVectorQuantumSystem
from qdes.utils_pyside import find_true_blocks_cpp, generate_circuit
from qdes.constants import (
    MAX_QUBITS, MAX_STEPS, THRESHOLD_ENTANGLED, THRESHOLD_NOT_ENTANGLED,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, TRAINING_CIRCUIT_COUNT,
    MIN_TRAINING_QUBITS, MAX_TRAINING_QUBITS, DATASET_VAL_FRAC,
    DATASET_TEST_FRAC, DATASET_SEED, SAMPLING_POINT,
    MODEL_D_MODEL, MODEL_N_HEAD, MODEL_NUM_LAYERS, MODEL_DROPOUT,
)

def _synced_shuffle(*arrays, seed):
    n = len(arrays[0])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return tuple([a[i] for i in perm] for a in arrays)

def generate_dataset(
    n_circuits=TRAINING_CIRCUIT_COUNT,
    max_qubits=MAX_QUBITS, max_steps=MAX_STEPS,
    val_frac = DATASET_VAL_FRAC, test_frac = DATASET_TEST_FRAC, 
    seed = DATASET_SEED, sampling_point = SAMPLING_POINT,
    out_path = "model_files/entanglement_dataset.npz"
):
    
    random.seed(seed)
    np.random.seed(seed)
    split_rng = np.random.RandomState(seed + 1)
    
    splits = {"train": [], "val": [], "test": []}
    split_outputs = {"train": [], "val": [], "test": []}
    split_probs = np.array([1 - val_frac - test_frac, val_frac, test_frac])

    for circuit_idx in range(n_circuits):
        split_name = split_rng.choice(["train", "val", "test"], p=split_probs)
        
        n_qubits = random.randint(MIN_TRAINING_QUBITS, MAX_TRAINING_QUBITS)
        n_ops = int(max_steps * (n_qubits / max_qubits))
        aliases = random.sample(range(max_qubits), n_qubits)
        
        sim = StateVectorQuantumSystem(n_qubits)
        
        # Start with an empty input matrix
        input_matrix = np.full((max_steps, 5), -1.0, dtype=np.float32)
        input_matrix[:, 3:] = 0.0

        execution_plan = generate_circuit(n_qubits, n_ops)

        for step, operation in enumerate(execution_plan):
            if operation[0] == "MEASURE":
                internal_target = operation[1]
                sim.measure(internal_target)
                input_matrix[step] = [
                    aliases[internal_target], aliases[internal_target],
                    OPERATION_IDS["MEASURE"], -1.0, -1.0
                ]
                k = 1
                op_name = "MEASURE"
            else:
                _, op_name, params, internal_targets = operation
                k = len(internal_targets)
                sim.apply(make_gate(op_name, *params), internal_targets)
                sp = compute_sp(op_name, *params)
                ep = compute_ep(op_name, *params)
                q1_alias = aliases[internal_targets[1]] if k > 1 else aliases[internal_targets[0]]
                input_matrix[step] = [
                    aliases[internal_targets[0]], q1_alias,
                    OPERATION_IDS[op_name], sp, ep
                ]

            # Control point
            if (step + 1) % sampling_point == 0 or (step + 1) == n_ops:
                sv, _ = sim.full_state()
                true_blocks = find_true_blocks_cpp(sv.copy(), n_qubits)
                
                output_matrix = np.zeros((max_qubits, max_qubits), dtype=np.int8)
                for block in true_blocks:
                    for i in range(len(block)):
                        alias_i = aliases[block[i]]
                        output_matrix[alias_i, alias_i] = 1
                        for j in range(i + 1, len(block)):
                            alias_j = aliases[block[j]]
                            output_matrix[alias_i, alias_j] = 1
                            output_matrix[alias_j, alias_i] = 1

                splits[split_name].append(input_matrix.copy())
                split_outputs[split_name].append(output_matrix.copy())

        # Print the progress
        if (circuit_idx + 1) % 50 == 0 or (circuit_idx + 1) == n_circuits:
            print(f"circuit: {circuit_idx + 1}/{n_circuits} completed.")

    # Shuffle
    result = {}
    for name in ["train", "val", "test"]:
        xs, ys = _synced_shuffle(splits[name], split_outputs[name], seed=seed + hash(name) % 1000)
        result[f"X_{name}"], result[f"y_{name}"] = np.stack(xs), np.stack(ys)

    # save
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, **result)
    
    return result

# PYTORCH Side

class EntanglementDataset(Dataset):
    #Converts quantum circuit data to PyTorch tensors.
    def __init__(self, X, y, max_qubits):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.triu_rows, self.triu_cols = np.triu_indices(max_qubits)

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X[idx]
        return {
            "q0": torch.from_numpy(row[:, 0].astype(np.int64) + 1),
            "q1": torch.from_numpy(row[:, 1].astype(np.int64) + 1),
            "op_id": torch.from_numpy(row[:, 2].astype(np.int64) + 1),
            "params": torch.from_numpy(row[:, 3:5].astype(np.float32)),
            "mask": torch.from_numpy((row[:, 2] != -1.0).astype(np.float32)),
            "target": torch.from_numpy(self.y[idx][self.triu_rows, self.triu_cols]),
        }

class EntanglementTransformer(nn.Module):
    def __init__(
        self, max_qubits, num_op_ids, d_model = MODEL_D_MODEL, 
        nhead = MODEL_N_HEAD, num_layers = MODEL_NUM_LAYERS, 
        max_steps = 80, dropout = MODEL_DROPOUT
    ):
        super().__init__()
        d_sub = d_model // 4
        
        # Embeddings
        self.q0_embed = nn.Embedding(max_qubits + 1, d_sub, padding_idx=0)
        self.q1_embed = nn.Embedding(max_qubits + 1, d_sub, padding_idx=0)
        self.op_embed = nn.Embedding(num_op_ids + 1, d_sub, padding_idx=0)
        
        # Lineer projections
        self.param_proj = nn.Linear(2, d_sub)
        self.input_proj = nn.Linear(d_sub * 4, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_steps, d_model) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model, max_qubits * (max_qubits + 1) // 2)
        )

    def forward(self, q0, q1, op_id, params, mask):
        x_concat = torch.cat([
            self.q0_embed(q0), self.q1_embed(q1),
            self.op_embed(op_id), self.param_proj(params)
        ], dim=-1)
        
        x = self.input_proj(x_concat) + self.pos_embed[:, :q0.size(1), :]
        
        x = self.transformer(x, src_key_padding_mask=(mask == 0))
        
        # Mean Pooling
        mask_f = mask.unsqueeze(-1)
        x_pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        
        return self.head(x_pooled)

# Training Loop

def run_epoch(model, loader, device, optimizer = None):

    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.BCEWithLogitsLoss()
    
    total_loss, total_elem, total_n = 0.0, 0, 0
    zone_not_entangled, zone_suspicious, zone_entangled = 0, 0, 0

    for batch in loader:
        # Transfer data to device (CPU/GPU)
        q0, q1, op_id = batch["q0"].to(device), batch["q1"].to(device), batch["op_id"].to(device)
        params, mask = batch["params"].to(device), batch["mask"].to(device)
        target = batch["target"].to(device)

        # Forward pass
        logits = model(q0, q1, op_id, params, mask)
        loss = loss_fn(logits, target)

        # Backward pass
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Statistical calculations
        probs = torch.sigmoid(logits)
        zone_not_entangled += (probs < THRESHOLD_NOT_ENTANGLED).sum().item()
        zone_suspicious += ((probs >= THRESHOLD_NOT_ENTANGLED) & (probs <= THRESHOLD_ENTANGLED)).sum().item()
        zone_entangled += (probs > THRESHOLD_ENTANGLED).sum().item()

        total_loss += loss.item() * target.size(0)
        total_elem += target.numel()
        total_n += target.size(0)

    return {
        "loss": total_loss / total_n,
        "z_04": zone_not_entangled / total_elem,
        "z_sup": zone_suspicious / total_elem,
        "z_06": zone_entangled / total_elem
    }

def train_model(
    data, max_qubits=MAX_QUBITS, epochs = DEFAULT_EPOCHS,
    batch_size = DEFAULT_BATCH_SIZE, lr = DEFAULT_LR,
    device = None, ckpt_path = "model_files/entanglement_model.pt"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        EntanglementDataset(data["X_train"], data["y_train"], max_qubits),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        EntanglementDataset(data["X_val"], data["y_val"], max_qubits),
        batch_size=batch_size, shuffle=False
    )

    model = EntanglementTransformer(
        max_qubits=max_qubits, num_op_ids=NUM_OP_IDS, max_steps=data["X_train"].shape[1]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        
        print(f"E{epoch}: TLoss: {train['loss']:0.4f} VLoss: {val['loss']:0.4f} "
              f"NotEnt(<{THRESHOLD_NOT_ENTANGLED}): {val['z_04']*100} "
              f"Suspicious({THRESHOLD_NOT_ENTANGLED}-{THRESHOLD_ENTANGLED}): {val['z_sup']*100} "
              f"Ent(>{THRESHOLD_ENTANGLED}): {val['z_06']*100}")
              
    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        
    torch.save(model.state_dict(), ckpt_path)
    return model

def export_to_onnx(
    model, max_qubits=MAX_QUBITS, max_steps=MAX_STEPS, num_op_ids=NUM_OP_IDS,
    onnx_path = "model_files/entanglement_model.onnx"
):
    model = model.to("cpu")
    model.eval()

    # Dummy inputs
    dummy_q0 = torch.randint(1, max_qubits + 1, (1, max_steps), dtype=torch.int64)
    dummy_q1 = torch.randint(1, max_qubits + 1, (1, max_steps), dtype=torch.int64)
    dummy_op = torch.randint(1, num_op_ids + 1, (1, max_steps), dtype=torch.int64)
    dummy_params = torch.randn(1, max_steps, 2, dtype=torch.float32)
    dummy_mask = torch.ones(1, max_steps, dtype=torch.float32)

    onnx_dir = os.path.dirname(onnx_path)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_q0, dummy_q1, dummy_op, dummy_params, dummy_mask),
        onnx_path,
        input_names=['q0', 'q1', 'op_id', 'params', 'mask'],
        output_names=['logits'],
        dynamic_axes={
            'q0': {0: 'batch_size'}, 'q1': {0: 'batch_size'},
            'op_id': {0: 'batch_size'}, 'params': {0: 'batch_size'},
            'mask': {0: 'batch_size'}, 'logits': {0: 'batch_size'}
        },
        opset_version=18,
        dynamo=False
    )

    print(f"Model has been exported to ONNX format at: {onnx_path}")


if __name__ == "__main__":
    if os.path.exists("model_files/entanglement_dataset.npz"):
        print("current dataset has been found and is being loaded...")
        data = dict(np.load("model_files/entanglement_dataset.npz"))
    else:
        print("dataset was not found, generating a new one...")
        data = generate_dataset()
        
    print("model training is starting...")
    model = train_model(data)
    
    print("ONNX export process is starting...")
    export_to_onnx(model)
    
    print("\nCompleted successfully.")