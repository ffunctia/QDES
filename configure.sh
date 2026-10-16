#!/usr/bin/env bash

set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" || "${VIRTUAL_ENV:-}" == "$(python3 -c 'import sys; print(sys.base_prefix)')" ]]; then
    echo "Error: configure.sh must be run inside an active virtualenv." >&2
    echo "Activate a virtualenv and run this script again." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OPTIONAL_PACKAGES=()
run_training=false

ask_yes_no() {
    local question="$1"
    local answer
    while true; do
        read -r -p "$question [y/n] " answer
        case "${answer,,}" in
            y|yes) return 0 ;;
            n|no) return 1 ;;
            *) echo "Please enter yes or no." ;;
        esac
    done
}

ask_choice() {
    local question="$1"
    local answer
    while true; do
        read -r -p "$question " answer
        if [[ "$answer" =~ ^[1-3]$ ]]; then
            printf '%s\n' "$answer"
            return 0
        fi
        echo "Please enter 1, 2, or 3."
    done
}

install_packages() {
    "$PYTHON_BIN" -m pip install --upgrade "$@"
}

echo "Starting QDES setup."
echo "Active virtualenv: ${VIRTUAL_ENV}"
install_packages -r requirements.txt
"$PYTHON_BIN" -m pip install --upgrade --editable .

if ask_yes_no "Did you change the supported qubit count?"; then
    rm -rf "$APP_ROOT/model_files"
    echo "A new dataset will be generated and the model will be retrained."
    run_training=true
else
    training_choice="$(ask_choice "Choose the model workflow: 1) Old data + old train 2) Old data + new train 3) New data + new train")"
    case "$training_choice" in
        1) ;;
        2) run_training=true ;;
        3)
            rm -rf "$APP_ROOT/model_files"
            run_training=true
            ;;
    esac
fi

if [[ "$run_training" == true ]]; then
    install_packages torch onnx
    OPTIONAL_PACKAGES+=(torch onnx)
    QDES_MODEL_PATH="$APP_ROOT/model_files/entanglement_model.onnx" \
        "$PYTHON_BIN" training/generate_dataset_and_train.py
fi

echo
echo "QDES setup and selected tasks completed."
if ((${#OPTIONAL_PACKAGES[@]} > 0)); then
    printf 'You may remove these optional packages manually later: %s\n' \
        "$(printf '%s ' "${OPTIONAL_PACKAGES[@]}" | xargs -n1 | sort -u | paste -sd' ' -)"
else
    echo "No optional packages were installed."
fi
echo "This script did not uninstall any packages."