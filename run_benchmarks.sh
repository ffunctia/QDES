#!/bin/bash

set -e

TESTS_PER_QUBIT="${1:-5}"
echo "Tests per qubit: $TESTS_PER_QUBIT"

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"
WORK_DIR="${APP_ROOT}/temp/circuits"
mkdir -p "$WORK_DIR"
TMP_DIR="${APP_ROOT}/temp/benchmarks"
mkdir -p "$TMP_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
export QDES_MODEL_PATH="${QDES_MODEL_PATH:-${APP_ROOT}/model_files/entanglement_model.onnx}"

echo "Test_no,n_qubits,n_gates,Max_er,average_ER,average_bm_ER,avg_gate_distance,n_multi_gates,max_depth,n_measurements" > "$TMP_DIR/temp_circuits.csv"
echo "QDES_speed_s,QDES_memory_MB" > "$TMP_DIR/temp_qdes.csv"
echo "SV_speed_s,SV_memory_MB" > "$TMP_DIR/temp_sv.csv"
echo "Quimb_speed_s,Quimb_memory_MB" > "$TMP_DIR/temp_quimb.csv"

QUBIT_CONFIGS=(16 20 24 28)
test_no=1

echo "Tests Producting..."
for n_qubits in "${QUBIT_CONFIGS[@]}"; do
    for (( t=1; t<=TESTS_PER_QUBIT; t++ )); do
        formatted_test_no=$(printf "%03d" "$test_no")
        pkl_file="$WORK_DIR/test_${formatted_test_no}_${n_qubits}q.pkl"

        echo "Circuiy Producting: Test ${formatted_test_no} (${n_qubits} Qubit)"

        circuit_line=$($PYTHON_BIN tests/controller.py "$test_no" "$n_qubits" "$WORK_DIR")
        echo "$circuit_line" >> "$TMP_DIR/temp_circuits.csv"

        n_gates=$(echo "$circuit_line" | cut -d',' -f3)
        echo "Running: Test ${formatted_test_no} (${n_qubits} Qubit, ${n_gates} Gates)"

        if ! $PYTHON_BIN tests/run_qdes.py "$pkl_file" "$n_qubits" >> "$TMP_DIR/temp_qdes.csv"; then
            echo "ERROR: run_qdes.py failed for Test ${formatted_test_no} (${n_qubits} Qubit)" >&2
            echo "nan,nan" >> "$TMP_DIR/temp_qdes.csv"
        fi

        if ! $PYTHON_BIN tests/run_sv.py "$pkl_file" "$n_qubits" >> "$TMP_DIR/temp_sv.csv"; then
            echo "ERROR: run_sv.py failed for Test ${formatted_test_no} (${n_qubits} Qubit)" >&2
            echo "nan,nan" >> "$TMP_DIR/temp_sv.csv"
        fi

        if ! $PYTHON_BIN tests/run_quimb.py "$pkl_file" "$n_qubits" >> "$TMP_DIR/temp_quimb.csv"; then
            echo "ERROR: run_quimb.py failed for Test ${formatted_test_no} (${n_qubits} Qubit)" >&2
            echo "nan,nan" >> "$TMP_DIR/temp_quimb.csv"
        fi

        rm -f "$pkl_file"

        test_no=$((test_no + 1))
    done
done

paste -d ',' "$TMP_DIR/temp_circuits.csv" "$TMP_DIR/temp_qdes.csv" "$TMP_DIR/temp_sv.csv" "$TMP_DIR/temp_quimb.csv" > final_performance_results.csv
rm -f "$TMP_DIR/temp_circuits.csv" "$TMP_DIR/temp_qdes.csv" "$TMP_DIR/temp_sv.csv" "$TMP_DIR/temp_quimb.csv"
rmdir "$TMP_DIR" 2>/dev/null || true
rm -rf "$WORK_DIR"

echo "Completed: 'final_performance_results.csv'"