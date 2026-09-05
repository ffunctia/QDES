#+
import numpy as np
from .constants import DTYPE_COMPLEX

# Projection Gatesseed
P0 = np.array([[1, 0], [0, 0]], dtype=DTYPE_COMPLEX)
P1 = np.array([[0, 0], [0, 1]], dtype=DTYPE_COMPLEX)

#Single-qubit fixed gates
I2 = np.eye(2, dtype=DTYPE_COMPLEX)

X = np.array([[0, 1], [1, 0]], dtype=DTYPE_COMPLEX)
Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE_COMPLEX)
Z = np.array([[1, 0], [0, -1]], dtype=DTYPE_COMPLEX)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=DTYPE_COMPLEX)
S = np.array([[1, 0], [0, 1j]], dtype=DTYPE_COMPLEX)
SDG = np.array([[1, 0], [0, -1j]], dtype=DTYPE_COMPLEX)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=DTYPE_COMPLEX)
TDG = np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]], dtype=DTYPE_COMPLEX)
SX = 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=DTYPE_COMPLEX)

# Single-qubit parametric gates
def RX(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=DTYPE_COMPLEX)

def RY(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=DTYPE_COMPLEX)

def RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=DTYPE_COMPLEX)

def P(phi):
    return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=DTYPE_COMPLEX)

def U3(theta, phi, lmbda):
    return np.array([
        [np.cos(theta/2), -np.exp(1j*lmbda)*np.sin(theta/2)],
        [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lmbda))*np.cos(theta/2)]
    ], dtype=DTYPE_COMPLEX)

# Two-qubit fixed gates
CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=DTYPE_COMPLEX)
CZ = np.diag([1,1,1,-1]).astype(DTYPE_COMPLEX)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=DTYPE_COMPLEX)

# Two-qubit parametric gates
def CP(phi):
    return np.diag([1,1,1,np.exp(1j*phi)]).astype(DTYPE_COMPLEX)

def CRX(theta):
    g = RX(theta)
    M = np.eye(4, dtype=DTYPE_COMPLEX)
    M[2:,2:] = g
    return M

def CRY(theta):
    g = RY(theta)
    M = np.eye(4, dtype=DTYPE_COMPLEX)
    M[2:,2:] = g
    return M

def CRZ(theta):
    g = RZ(theta)
    M = np.eye(4, dtype=DTYPE_COMPLEX)
    M[2:,2:] = g
    return M

def RXX(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,0,-1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[-1j*s,0,0,c]], dtype=DTYPE_COMPLEX)

def RYY(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,0,1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[1j*s,0,0,c]], dtype=DTYPE_COMPLEX)

def RZZ(theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2),
                    np.exp(1j*theta/2), np.exp(-1j*theta/2)]).astype(DTYPE_COMPLEX)

class GateMetadata:
    def __init__(self, name, num_qubits, is_parametric=False, sp_func=None, ep_func=None):
        self.name = name
        self.num_qubits = num_qubits
        self.is_parametric = is_parametric
        self.sp_func = sp_func
        self.ep_func = ep_func

def _const_sp(c): return lambda *args: c
def _const_ep(c): return lambda *args: c
def _sp_sin_half(theta): return abs(np.sin(theta/2))
def _sp_sin(theta): return abs(np.sin(theta))
def _ep_sin_half(theta): return abs(np.sin(theta/2))
def _ep_sin(theta): return abs(np.sin(theta))

GATE_METADATA = {
    # Single fixed
    'X': GateMetadata('X', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'Y': GateMetadata('Y', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'Z': GateMetadata('Z', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'H': GateMetadata('H', 1, sp_func=_const_sp(1.0), ep_func=_const_ep(0.0)),
    'S': GateMetadata('S', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'SDG': GateMetadata('SDG', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'T': GateMetadata('T', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'TDG': GateMetadata('TDG', 1, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'SX': GateMetadata('SX', 1, sp_func=_const_sp(1.0), ep_func=_const_ep(0.0)),
    # Two fixed
    'CX': GateMetadata('CX', 2, sp_func=_const_sp(0.0), ep_func=_const_ep(1.0)),
    'CZ': GateMetadata('CZ', 2, sp_func=_const_sp(0.0), ep_func=_const_ep(1.0)),
    'SWAP': GateMetadata('SWAP', 2, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    # Single parametric
    'RX': GateMetadata('RX', 1, is_parametric=True, sp_func=_sp_sin, ep_func=_const_ep(0.0)),
    'RY': GateMetadata('RY', 1, is_parametric=True, sp_func=_sp_sin, ep_func=_const_ep(0.0)),
    'RZ': GateMetadata('RZ', 1, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'P': GateMetadata('P', 1, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_const_ep(0.0)),
    'U3': GateMetadata('U3', 1, is_parametric=True, sp_func=lambda t,p,l: _sp_sin_half(t), ep_func=_const_ep(0.0)),
    # Two parametric
    'CP': GateMetadata('CP', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin_half),
    'CRX': GateMetadata('CRX', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin_half),
    'CRY': GateMetadata('CRY', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin_half),
    'CRZ': GateMetadata('CRZ', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin_half),
    'RXX': GateMetadata('RXX', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin),
    'RYY': GateMetadata('RYY', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin),
    'RZZ': GateMetadata('RZZ', 2, is_parametric=True, sp_func=_const_sp(0.0), ep_func=_ep_sin),
}

def compute_sp(name, *params):
    md = GATE_METADATA.get(name)
    if md is None:
        return -1.0
    return md.sp_func(*params) if md.sp_func else 0.0

def compute_ep(name, *params):
    md = GATE_METADATA.get(name)
    if md is None:
        return -1.0
    return md.ep_func(*params) if md.ep_func else 0.0

def make_gate(name, *params):
    gate = globals().get(name)
    if gate is None or (params and not callable(gate)) or (not params and callable(gate)):
        raise ValueError(f"Unknown gate: {name}")
    return gate(*params) if params else gate

def infer_metadata(name, gate):
    if name in GATE_METADATA and not GATE_METADATA[name].is_parametric:
        return compute_sp(name), compute_ep(name)

    matrix = np.asarray(gate)
    if name in ('RX', 'RY'):
        sp = 2.0 * abs(matrix[0, 1]) * abs(matrix[0, 0])
        return float(min(1.0, sp)), 0.0
    if name == 'U3':
        return float(abs(matrix[1, 0])), 0.0
    if name in ('CP', 'CRZ'):
        return 0.0, float(abs(np.sin(np.angle(matrix[3, 3]) / 2.0)))
    if name in ('CRX', 'CRY'):
        return 0.0, float(abs(matrix[2, 3]))
    if name in ('RXX', 'RYY'):
        return 0.0, float(min(1.0, 2.0 * abs(matrix[0, 3])))
    if name == 'RZZ':
        phase = np.angle(matrix[0, 0])
        return 0.0, float(abs(np.sin(2.0 * phase)))
    return 0.0, 0.0

OPERATION_IDS = {"MEASURE": 0}
_gid = 1
for name in ['X','Y','Z','H','S','SDG','T','TDG','SX','CX','CZ','SWAP']:
    OPERATION_IDS[name] = _gid; _gid += 1
for name in ['RX','RY','RZ','P','U3','CP','CRX','CRY','CRZ','RXX','RYY','RZZ']:
    OPERATION_IDS[name] = _gid; _gid += 1
NUM_OP_IDS = _gid