# ML-QDES

**ML-QDES** means **Machine Learning-based Quantum Dynamic
Entanglement-aware Simulator**. It is a Python quantum-circuit simulator with
a C++/OpenMP numerical backend, andoptional ONNX-assisted entanglement partitioning.

ML-QDES evolves pure quantum states and offers a grouped simulator that can
split separable subsystems into smaller state vectors. This can reduce the
amount of state-vector work for circuits whose qubits remain weakly entangled.
The project also includes state-vector fidelity checks, benchmark runners, and
optional model-training utilities.

## Contents

- [Highlights](#highlights)
- [How it fits together](#how-it-fits-together)
- [Requirements and installation](#requirements-and-installation)
- [Quick start](#quick-start)
- [Simulators and API](#simulators-and-api)
- [Supported gates](#supported-gates)
- [Testing and validation](#testing-and-validation)
- [Benchmarks](#benchmarks)
- [Optional entanglement model](#optional-entanglement-model)
- [Repository map](#repository-map)
- [License and authorship](#license-and-authorship)

## Requirements and installation

### Requirements

- Python **3.9 or newer**
- NumPy **1.23 or newer**
- A C++14 compiler and the Python/NumPy/pybind11 build headers
- OpenMP support for the default non-Windows extension build

The base package builds its `qdes.utils_cside` extension during installation.
Use a virtual environment so the build and runtime dependencies are isolated:

```bash
git clone https://github.com/ffunctia/QDES.git
cd QDES
VENV_DIR="my-ml-qdes-env"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
```

Choose any environment directory name you prefer in `VENV_DIR`; the example
uses `my-ml-qdes-env`. For editable development installs, `-e .` keeps Python
imports pointed at the checkout. For a regular install, use
`python -m pip install .`.

### Optional development and validation dependencies

Install the declared test extras to run the fidelity checks:

```bash
python -m pip install -e ".[test]"
```

The repository's `requirements-tests.txt` also includes `kahypar`, used by
some reference/utility workflows:

```bash
python -m pip install -r requirements-tests.txt
```

`torch` and `onnx` are only needed to train or regenerate the optional model.
They are not required for ordinary simulation.

### Alternative setup script

`configure.sh` automates installation inside whichever virtual environment is
already activated and prompts about model/data workflows. Read its prompts
carefully: some choices remove and regenerate `model_files/` and can start a
long training run. Manual installation above is the non-interactive core setup.

### CMake build

The extension can also be built with CMake when Python development headers,
NumPy, pybind11, a C++14 compiler, and (optionally) OpenMP are available:

```bash
cmake -S . -B build
cmake --build build --parallel
```

The resulting module is placed under `build/qdes/`. For the usual Python
installation workflow, `python -m pip install .` builds and installs the
extension into the active environment.

## Testing and validation

Run commands from the repository root. Setting `PYTHONPATH` explicitly makes
the tests import this checkout rather than a different installed `qdes`
package:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_sampling.py
PYTHONPATH="$PWD/src" python tests/uni_test.py 100 42 5
```

The first command checks the joint sampling distribution of two independent
Hadamard qubits and verifies deterministic behavior for a fixed seed. The
second runs interleaved grouped-simulator/state-vector fidelity and grouping
checks; its positional arguments are number of tests, seed, and qubit count.
The current `uni_test.py` accepts the seed argument but does not reseed its
random circuit generator from it, so this command should not be treated as
producing an identical circuit sequence on every run.

Quimb-based state-vector comparisons use 8-qubit, 122-operation circuits and
run 1000 cases by default. They can be launched separately:

```bash
PYTHONPATH="$PWD/src" python tests/sv_quimb_fidelity.py 42
PYTHONPATH="$PWD/src" python tests/sv_gf_quimb_fidelity.py 42
```

These longer fidelity suites are separate from the focused sampling regression
test. Quimb must be installed first; see [Optional development and validation
dependencies](#optional-development-and-validation-dependencies).

## Benchmarks

The benchmark compares ML-QDES, the gate-fusion state-vector simulator, and
Quimb on generated circuits. It reports circuit/grouping metrics, elapsed time,
and peak resident memory; it is a performance runner, not a
sampling-distribution correctness test.

```bash
# Generate one circuit for each configured qubit size (16, 20, 24, 28)
bash run_benchmarks.sh 1

# Use the default: 25 circuits per configured qubit size
bash run_benchmarks.sh
```

The script uses the active `python3` by default. Override it with
`PYTHON_BIN=/path/to/python bash run_benchmarks.sh 1` if needed. The optional
Quimb benchmark requires Quimb to be installed.

Generated circuit files are placed temporarily in `temp/circuits/`, temporary
benchmark data in `temp/benchmarks/`, and the combined output is written to
`final_performance_results.csv` in the repository root. The temporary circuit
and benchmark files are removed at the end. Larger qubit configurations can
take substantial time and memory; begin with a small test count.

## Repository map

```text
ML-QDES/
├── csrc/                 C++/OpenMP numerical extension
├── model_files/          Optional trained model assets
├── src/qdes/             Python package and simulator APIs
├── tests/                Fidelity, sampling, and benchmark runners
├── training/             Dataset generation and model training
├── configure.sh          Interactive environment/model setup
└── run_benchmarks.sh     Circuit generation and performance runner
```