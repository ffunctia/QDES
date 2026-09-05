# QDES

QDES is a quantum circuit simulator with an optional C++/OpenMP acceleration
module and optional machine-learning and reference-simulator tooling.

## Quick setup

The core installation only needs NumPy and the C++ build prerequisites:

```bash
source .venv/bin/activate
bash configure.sh
```

The script uses the already-active virtualenv and installs current package
versions with `pip`. It asks whether the supported qubit count changed, then
offers these model workflows: reuse the old data and old model, reuse the old
data and retrain, or generate new data and retrain. It asks separately about QDES,
state-vector fidelity, and benchmark tests, installing `quimb` only when it is
needed. It never uninstalls packages; at the end it lists optional packages
that can be removed manually.

## Manual installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

The `tests/` directory is a developer/validation area and is not installed as
part of the QDES package. A clone that contains LFS pointers must
run `git lfs install` and `git lfs pull` before using the bundled model.

The validation entry point is `tests/uni_test.py`. Dataset generation and
training are kept separately in `training/generate_dataset_and_train.py`.
Benchmark circuits are generated temporarily under `temp/circuits/` and are
removed when the benchmark finishes.

After installing with `pip install .`, run benchmarks from the repository root
so the bundled model can be discovered:

```bash
python tests/uni_test.py 100 42 5
bash run_benchmarks.sh 1
```

The benchmark uses the installed `qdes` package, including its compiled C++
extension. To use a model from another location, set `QDES_MODEL_PATH` to the
ONNX file before running a command.

## CMake build

The same C++ source can be built independently when Python, NumPy, pybind11,
and a C++14 compiler are available:

```bash
cmake -S . -B build
cmake --build build --parallel
```

The module is emitted under `build/qdes/`. `pip install .` is the recommended
way to build and install it into the active Python environment.
