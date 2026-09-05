from numpy import complex128, int64, array, random

#Types
DTYPE_COMPLEX = complex128
DTYPE_INT = int64

#States
STATE_ZERO = array([1.0 + 0j, 0.0 + 0j], dtype=DTYPE_COMPLEX)
STATE_ONE  = array([0.0 + 0j, 1.0 + 0j], dtype=DTYPE_COMPLEX)

#Main simulation limits
MAX_QUBITS = 28
MAX_STEPS = MAX_QUBITS * 16

#Main simulation parameters
BRANCH_DEPTH = 2
THRESHOLD_ENTANGLED = 0.9
THRESHOLD_NOT_ENTANGLED = 0.1

#Fusion limits
MAX_FUSION_QUBITS = 6

#Tolerance and Epsilon values PYTHON side
PYTHON_NORM_EPS = 1e-12
TEST_FIDELITY_TOL = 1e-5

# Tolerance and Epsilon values C++ side
CPP_FACTOR_ENTANGLED_EPS = 1e-8
CPP_ENTANGLED_TOL = 1e-8
CPP_NORM_EPS = 1e-300

#OpenMP chunk size for entangled state simulation in C++
CPP_ENTANGLED_CHUNK = 256

#Training and tests operation probabilities
RANDOM_OPERATION_PROBABILITIES = (0.05, 0.285, 0.19, 0.285, 0.19)

# Training and dataset defaults
DEFAULT_EPOCHS = 30
TRAINING_CIRCUIT_COUNT = 120_000
MIN_TRAINING_QUBITS = 2
MAX_TRAINING_QUBITS = min(MAX_QUBITS, 14)

DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3

# Dataset generation defaults
DATASET_VAL_FRAC = 0.1
DATASET_TEST_FRAC = 0.1
SAMPLING_POINT = 30

# Model defaults
MODEL_D_MODEL = 64
MODEL_N_HEAD = 4
MODEL_NUM_LAYERS = 3
MODEL_DROPOUT = 0.1

#Gate definitions
RANDOM_PARAM_SINGLE_QUBIT_GATES = ('RX', 'RY', 'RZ', 'P', 'U3')
RANDOM_PARAM_TWO_QUBIT_GATES = ('CP', 'CRX', 'CRY', 'CRZ', 'RXX', 'RYY', 'RZZ')
RANDOM_SINGLE_FIXED_NAMES = ('X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG', 'SX')
RANDOM_TWO_FIXED_NAMES = ('CX', 'CZ', 'SWAP')

#SEED PARAMETERS
DATASET_SEED = 42
DEFAULT_TEST_SEED = 42

#short functions
CALCULATE_STEPS = lambda n_qubits: int(MAX_STEPS / MAX_QUBITS * n_qubits)
RAND_SEED = lambda: int(random.randint(0, 2**31 - 1))
